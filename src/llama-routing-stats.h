#pragma once

// Routing statistics: while serving, record how many tokens each expert of each mixture-of-experts
// layer received. Trained routing is skewed, how skewed depends on the weights, and nothing in a
// GGUF header says by how much -- so a cost model for a mixture of experts has to measure it. This
// records it from real traffic instead of a benchmark.
//
// LLAMA_ROUTING_STATS selects how often a micro-batch is recorded:
//
//   unset / "off"   off, and nothing below runs
//   "on"            at most one micro-batch per second
//   <milliseconds>  at most one micro-batch per that many; "0" records every micro-batch
//
// Nothing about the tokens themselves is kept -- not the ids, not the text, not the position. What
// is kept is a count per expert per layer, plus how many micro-batches and tokens went into it.
//
// Prefill and decode are counted separately, and each has its own timer. They are different token
// distributions, and what routing does with one says little about the other; keeping one histogram
// would also have starved it, because a period that samples a decode-heavy hour collects one token
// a second, where a single prefill micro-batch carries five hundred for the same overhead.
//
// The cost is paid only by a recorded micro-batch. ggml's eval callback splits a graph at every
// node the callback asks for and synchronizes between the pieces, so the callback is installed
// around one compute call and removed again, rather than left on the scheduler.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <mutex>
#include <vector>

struct llama_routing_recorder {
    // a micro-batch of more than one token, and one of exactly one: prefill and decode in practice
    enum kind { PREFILL = 0, DECODE = 1, N_KIND = 2 };

    // configuration, fixed once
    bool    on        = false;
    int64_t period_us = 1000000;

    // set from the model's header when the context is created
    int32_t n_expert      = 0;
    int32_t n_expert_used = 0;

    mutable std::mutex mu;

    struct layer {
        std::vector<int64_t> counts[N_KIND];   // tokens routed to each of the n_expert experts
        int64_t              n_tokens[N_KIND] = {0, 0};

        // Summaries of each micro-batch on its own, summed so a mean survives without keeping
        // every micro-batch. They are not recoverable from the pooled counts above and are the
        // ones a cost model wants: a layer's popularity is redrawn per micro-batch, so pooling
        // many of them averages that away and makes routing look more even than it is. Measured
        // on lfm2.5:8b over seven micro-batches, the busiest expert was 6.2 times an even share
        // per micro-batch and 4.9 times it pooled.
        //
        // Each is weighted by the micro-batch's tokens, not counted once: a two-token warmup
        // batch and a 512-token prefill are the same row otherwise, and their shapes are not
        // comparable. Where every micro-batch is the same size this is the plain mean.
        // For decode these three carry nothing: one token picks k distinct experts, so touched is
        // always k/E, effective experts always k, and busiest always E/k. Decode's pooled counts
        // are the informative half -- they are what many single tokens did between them.
        int64_t n_batches[N_KIND]   = {0, 0};
        double  sum_weight[N_KIND]  = {0, 0}; // tokens behind the three sums below
        double  sum_touched[N_KIND] = {0, 0}; // share of experts that got at least one token
        double  sum_eff[N_KIND]     = {0, 0}; // effective experts, (sum c)^2 / sum c^2
        double  sum_busiest[N_KIND] = {0, 0}; // busiest expert's tokens over an even share
    };

    // The last layer of a model routes only the tokens whose output is needed: llama.cpp puts a
    // ggml_get_rows on the output ids ahead of the last block's feed-forward, so during prefill its
    // router sees one token where every other layer sees the whole micro-batch. Its counts are a
    // real sample of a different population -- so every layer carries its own token totals, and a
    // consumer divides by those rather than by the totals below.
    std::map<int32_t, layer> layers;

    int64_t n_ubatch_seen              = 0; // micro-batches offered, recorded or not
    int64_t n_ubatch_recorded[N_KIND]  = {0, 0};
    int64_t n_tokens[N_KIND]           = {0, 0};
    int64_t t_read_us                  = 0; // time spent reading routing back off the devices

    // staging for the read-back and this micro-batch's own histogram, touched only by the thread
    // running the graph
    std::vector<int32_t> staging;
    std::vector<int64_t> batch;

    static bool parse_env(int64_t * period_us) {
        const char * v = getenv("LLAMA_ROUTING_STATS");
        if (v == nullptr || *v == '\0' || strcmp(v, "off") == 0) {
            return false;
        }
        if (strcmp(v, "on") == 0) {
            *period_us = 1000000;
            return true;
        }
        char * end = nullptr;
        const long long ms = strtoll(v, &end, 10);
        if (end == v || *end != '\0' || ms < 0) {
            return false;
        }
        *period_us = (int64_t) ms * 1000;
        return true;
    }

    // Called when a context is created. The recorder is one per process, not one per context: a
    // context is freed when the server sleeps, while a reader may be asking at that moment, and the
    // counts are worth more than the lifetime coupling. ollama runs one llama-server per model, so
    // per process is per model there. A context for a model with a different number of experts
    // clears what is held rather than adding to it.
    void configure(int32_t n_expert_, int32_t n_expert_used_) {
        std::lock_guard<std::mutex> lock(mu);
        on = parse_env(&period_us);
        if (n_expert_ <= 0) {
            on = false; // not a mixture of experts
        }
        if (n_expert_ != n_expert) {
            clear();
        }
        n_expert      = n_expert_;
        n_expert_used = n_expert_used_;
    }

    // Decide whether this micro-batch is recorded. `taken` says the scheduler's eval callback
    // belongs to someone else (the offline routing harness sets one); theirs wins and we stand down
    // rather than fight over a single pointer.
    bool begin(bool batched, bool taken) {
        if (!on) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mu);
        n_ubatch_seen++;
        if (taken) {
            return false;
        }
        const int k   = batched ? PREFILL : DECODE;
        const int64_t now = ggml_time_us();
        if (period_us > 0 && now - t_last_us[k] < period_us) {
            return false;
        }
        t_last_us[k] = now;
        recording    = k;
        return true;
    }

    void end() {
        std::lock_guard<std::mutex> lock(mu);
        n_ubatch_recorded[recording]++;
    }

    // called from the scheduler's eval callback, on the thread running the graph
    void record(int32_t il, const ggml_tensor * t) {
        if (t->type != GGML_TYPE_I32 || t->ne[0] <= 0 || t->ne[1] <= 0) {
            return;
        }
        const int64_t t0 = ggml_time_us();

        // [n_expert_used, n_tokens] I32, usually a strided view of the argsort. One read of the
        // whole span beats a read per token: the span is the argsort, a few hundred KiB.
        const size_t span = ggml_nbytes(t);
        staging.resize((span + sizeof(int32_t) - 1) / sizeof(int32_t));
        ggml_backend_tensor_get(t, staging.data(), 0, span);

        const int64_t k      = t->ne[0];
        const int64_t n      = t->ne[1];
        const size_t  stride = t->nb[1] / sizeof(int32_t);

        // this micro-batch on its own first, so its shape can be summarised before it is pooled
        batch.assign(n_expert, 0);
        for (int64_t i = 0; i < n; i++) {
            const int32_t * row = staging.data() + (size_t) i * stride;
            for (int64_t j = 0; j < k; j++) {
                const int32_t e = row[j];
                if (e >= 0 && e < n_expert) {
                    batch[e]++;
                }
            }
        }
        double total = 0, sq = 0, busiest = 0, touched = 0;
        for (int32_t e = 0; e < n_expert; e++) {
            const double v = (double) batch[e];
            total += v;
            sq    += v * v;
            busiest = std::max(busiest, v);
            touched += v > 0 ? 1 : 0;
        }

        std::lock_guard<std::mutex> lock(mu);
        // the whole micro-batch is one kind, even where the last layer sees fewer of its tokens
        const int which = recording;
        auto & lay = layers[il];
        auto & c   = lay.counts[which];
        if ((int32_t) c.size() != n_expert) {
            c.assign(n_expert, 0);
        }
        for (int32_t e = 0; e < n_expert; e++) {
            c[e] += batch[e];
        }
        if (total > 0) {
            const double w = (double) n;
            lay.n_batches[which]++;
            lay.sum_weight[which]  += w;
            lay.sum_touched[which] += w * (touched / n_expert);
            lay.sum_eff[which]     += w * (total * total / sq);
            lay.sum_busiest[which] += w * (busiest / (total / n_expert));
        }
        lay.n_tokens[which] += n;
        // the micro-batch's own tokens are counted once, on whichever layer routes first
        if (il == layers.begin()->first) {
            n_tokens[which] += n;
        }
        t_read_us += ggml_time_us() - t0;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mu);
        clear();
    }

private:
    int64_t t_last_us[N_KIND] = {0, 0};
    int     recording         = PREFILL; // the kind of the micro-batch being recorded

    void clear() { // call with mu held
        layers.clear();
        n_ubatch_seen = 0;
        t_read_us     = 0;
        for (int k = 0; k < N_KIND; k++) {
            n_ubatch_recorded[k] = 0;
            n_tokens[k]          = 0;
        }
    }
};

// one per process; see configure()
inline llama_routing_recorder & llama_routing_stats_recorder() {
    static llama_routing_recorder rec;
    return rec;
}

// the scheduler's eval callback while a micro-batch is being recorded
inline bool llama_routing_stats_eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * rec = (llama_routing_recorder *) user_data;
    static const char prefix[] = "ffn_moe_topk-";
    if (strncmp(t->name, prefix, sizeof(prefix) - 1) != 0) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    rec->record((int32_t) atoi(t->name + sizeof(prefix) - 1), t);
    return true;
}
