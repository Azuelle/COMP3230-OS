/*
 * PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
 * FILE NAME: parallel_3035974119.c
 * NAME: TANG Jiakai
 * UID: 3035974119
 * Development Platform: gcc version 10.2.1
 *                       Debian 10.2.1-6
 *                       (in Docker container on macOS)
 * Remark: (How much you implemented?) Everything
 * How to compile separately:
 *      gcc -o parallel parallel_3035974119.c -O2 -lm -lpthread
 */

#include "common.h"  // some common definitions

#include <unistd.h>        // for nearly everything :)
#include <stdio.h>         // for printf, sprintf, fgets
#include <stdlib.h>        // for malloc, calloc
#include <stdint.h>        // for uint8_t and uint64_t
#include <time.h>          // for time
#include <string.h>        // for memcpy and strcmp
#include <sys/resource.h>  // for rusage collection

#include "model.h"  // for Llama definitions -> no need to know

int pos = 0;              // global position of generation
Transformer transformer;  // transformer instance to be init
Tokenizer tokenizer;      // tokenizer instance to be init
Sampler sampler;          // sampler instance to be init

// YOUR CODE STARTS HERE
#include <pthread.h>
#include <semaphore.h>
// #include <stdbool.h>   // uncomment this line if you want true / false

// you may define global variables here

typedef struct rusage rusage;

int num_threads;
pthread_t* threads;

sem_t* task_sem;
sem_t* done_sem;

typedef enum TaskType { MAT_VEC_MUL, MULTI_HEAD_ATTN, TERMINATE } TaskType;
TaskType task;

typedef struct MatVecMulArgs {
    int col;
    int row;
    float* out;
    QuantizedTensor* vec;
    QuantizedTensor* mat;
} MatVecMulArgs;
typedef struct MultiHeadAttnArgs {
    int seq_len;
    int n_heads;
    int head_size;
    int kv_dim;
    int kv_mul;
    float* out;
    float* q;
    float* key_cache;
    float* value_cache;
    float* att;
} MultiHeadAttnArgs;
MatVecMulArgs mat_vec_mul_args;
MultiHeadAttnArgs multi_head_attn_args;

rusage* usages;

// function executed by each thread to complete mat_vec_mul
// @note: please modify the signature to what you want
void mat_vec_mul_task_func(int id, MatVecMulArgs args) {
    // extract arguments
    float* out = args.out;
    QuantizedTensor* vec = args.vec;
    QuantizedTensor* mat = args.mat;
    int col = args.col;
    int row = args.row;

    // for each row
    // this thread is responsible for calculating
    // id*(row/num_threads) to (id+1)*(row/num_threads)
    int workload = row / num_threads;
    int from = id * workload;
    int to = (id + 1) * workload;
    if (id == num_threads - 1) to = row;
    // fprintf(stderr, "Thread %d - Row %d~%d/%d\n", id, from, to, row);

    for (int i = from; i < to; i++) {
        float val = 0.0f;  // final value
        int32_t ival = 0;  // integer value to be dequantized
        int in = i * col;  //

        // for each column
        // GS is group size of quantization, not included in assignment
        // @note please don't parallel this loop
        for (int j = 0; j <= col - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival +=
                    ((int32_t)vec->q[j + k]) * ((int32_t)mat->q[in + j + k]);
            }
            val += ((float)ival) * mat->s[(in + j) / GS] * vec->s[j / GS];
            ival = 0;
        }
        out[i] = val;
    }
}

// function executed by each thread to complete multi_head_attn
// @note: please modify the signature to what you want
void multi_head_attn_task_func(int id, MultiHeadAttnArgs args) {
    // extract arguments
    float* out = args.out;              // output tensor [head, head_size]
    float* q = args.q;                  // query tensor  [head, head_size]
    float* key_cache = args.key_cache;  // cache of history key tensor [kv_head,
                                        // seq_len, head_size]
    float* value_cache = args.value_cache;  // cache of history value tensor
                                            // [kv_head, seq_len, head_size]
    float* att = args.att;       // buffer for attention score [head, seq_len]
    int seq_len = args.seq_len;  // current sequence length
    int n_heads = args.n_heads;  // number of heades
    int head_size = args.head_size;  // size of each head
    int kv_dim = args.kv_dim;
    int kv_mul = args.kv_mul;

    // multihead attention. iterate over all heads
    // this thread is responsible for calculating
    // id*(n_heads/num_threads) to (id+1)*(n_head/num_threads)
    int workload = n_heads / num_threads;
    int from = id * workload;
    int to = (id + 1) * workload;
    if (id == num_threads - 1) to = n_heads;
    // fprintf(stderr, "Thread %d - Head %d~%d/%d\n", id, from, to, n_heads);

    for (int h = from; h < to; h++) {
        // get the query vector for this head
        float* head_q = q + h * head_size;
        // attention scores for this head
        float* head_att = att + h * seq_len;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* head_k = key_cache + t * kv_dim + (h / kv_mul) * head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += head_q[i] * head_k[i];
            }
            score /= sqrtf(head_size);
            // save the score to the attention buffer
            head_att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(head_att, pos + 1);

        // weighted sum of the values, store back into xb
        float* head_out = out + h * head_size;
        memset(head_out, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float* head_v = value_cache + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = head_att[t];
            // accumulate the weighted value into head out
            for (int i = 0; i < head_size; i++) {
                head_out[i] += a * head_v[i];
            }
        }
    }
}

// thread function used in pthread_create
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void* thr_func(void* arg) {
    int id = *(int*)arg;
    // fprintf(stderr, "Hi from %d\n", id);
    sem_post(&done_sem[id]);  // signal ready

    while (1) {
        sem_wait(&task_sem[id]);
        // fprintf(stderr, "Thread %d is working\n", id);
        switch (task) {
            case TERMINATE:
                getrusage(RUSAGE_THREAD, &usages[id]);
                // fprintf(stderr, "Bye from %d\n", id);
                sem_post(&done_sem[id]);
                return NULL;

            case MAT_VEC_MUL:
                mat_vec_mul_task_func(id, mat_vec_mul_args);
                break;
            case MULTI_HEAD_ATTN:
                multi_head_attn_task_func(id, multi_head_attn_args);
                break;
        }
        sem_post(&done_sem[id]);
    }
}

// function to initialize thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void init_thr_pool(int num_thr) {
    num_threads = num_thr;
    threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    usages = (rusage*)malloc(num_threads * sizeof(rusage));
    task_sem = (sem_t*)malloc(num_threads * sizeof(sem_t));
    done_sem = (sem_t*)malloc(num_threads * sizeof(sem_t));

    for (int i = 0; i < num_threads; i++) {
        sem_init(&task_sem[i], 0, 0);
        sem_init(&done_sem[i], 0, 0);
        // fprintf(stderr, "Initialized sems for thread %d\n", i);
        pthread_create(&threads[i], NULL, thr_func, &i);
        sem_wait(&done_sem[i]);  // wait for thread to be ready
    }
}

inline double get_time(struct timeval t) {
    return t.tv_sec + t.tv_usec / 1000000.0;
}

// function to close thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void close_thr_pool() {
    task = TERMINATE;
    for (int i = 0; i < num_threads; i++) {
        sem_post(&task_sem[i]);
        pthread_join(threads[i], NULL);
        sem_wait(&done_sem[i]);

        printf(
            "\033[0;32mThread %d has terminated - user %.4f s, system %.4f s\n",
            i, get_time(usages[i].ru_utime), get_time(usages[i].ru_stime));

        sem_destroy(&task_sem[i]);
        sem_destroy(&done_sem[i]);
    }
    free(threads);
    free(usages);
    free(task_sem);
    free(done_sem);

    rusage main_thr_usage;
    getrusage(RUSAGE_THREAD, &main_thr_usage);
    printf("Main thread - user %.4fs, system %.4fs\n",
           get_time(main_thr_usage.ru_utime),
           get_time(main_thr_usage.ru_stime));
    rusage process_usage;
    getrusage(RUSAGE_SELF, &process_usage);
    printf("Process - user %.4fs, system %.4fs\033[0m\n",
           get_time(process_usage.ru_utime), get_time(process_usage.ru_stime));
}

void broadcast_and_do_task() {
    // fprintf(stderr, "Broadcasting task\n");
    for (int i = 0; i < num_threads; i++) sem_post(&task_sem[i]);
    // fprintf(stderr, "Waiting for threads to finish\n");
    for (int i = 0; i < num_threads; i++) sem_wait(&done_sem[i]);
    // fprintf(stderr, "Finished\n");
}

// ----------------------------------------------------------------------------
// entry function for multi-threading matrix multiplication
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void mat_vec_mul(float* out, QuantizedTensor* vec, QuantizedTensor* mat,
                 int col, int row) {
    MatVecMulArgs args = {col, row, out, vec, mat};
    mat_vec_mul_args = args;
    task = MAT_VEC_MUL;
    broadcast_and_do_task();
}

// ----------------------------------------------------------------------------
// entry function for multi-threading multi-head-attention
// @note: YOU CAN NOT MODIFY FUNCTION SIGNATURE!!!
void multi_head_attn(float* out,        // output tensor [head, head_size]
                     float* q,          // query tensor  [head, head_size]
                     float* key_cache,  // cache of history key tensor [kv_head,
                                        // seq_len, head_size]
                     float* value_cache,  // cache of history value tensor
                                          // [kv_head, seq_len, head_size]
                     float* att,  // buffer for attention score [head, seq_len]
                     int seq_len, int n_heads, int head_size, int kv_dim,
                     int kv_mul) {
    MultiHeadAttnArgs args = {seq_len, n_heads, head_size, kv_dim,      kv_mul,
                              out,     q,       key_cache, value_cache, att};
    multi_head_attn_args = args;
    task = MULTI_HEAD_ATTN;
    broadcast_and_do_task();
}
// YOUR CODE ENDS HERE

// ----------------------------------------------------------------------------
// forward Transformer, you're not allowed to modify this part
float* forward(Transformer* transformer, int token, int pos) {
    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul =
        p->n_heads /
        p->n_kv_heads;  // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->q, &s->xq, w->wq + l, dim, dim);
        mat_vec_mul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        mat_vec_mul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in
        // each head
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn =
                i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0
                                 ? s->q
                                 : s->k;  // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff =
            l * p->seq_len * kv_dim;  // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        multi_head_attn(s->xb, s->q, s->key_cache + loff, s->value_cache + loff,
                        s->att, p->seq_len, p->n_heads, head_size, kv_dim,
                        kv_mul);

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) *
        // self.w3(x)) first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        mat_vec_mul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        mat_vec_mul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    mat_vec_mul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// generation loop, you're not allowed to modify this part
void generate(char* prompt) {
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(
        (strlen(prompt) + 6) * sizeof(int));  // +6 reserved for prompt template
    encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr,
                "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;                      // place holder for next token
    int token = prompt_tokens[0];  // place holder of prev token, kickoff as
                                   // prompt_tokens[0]
    int end_pos = pos + MAX_NEW_TOKENS + num_prompt_tokens;
    int start_pos = pos;
    long start_time = 0;  // to be lazy iniialzied
    while (pos < end_pos) {
        // forward the transformer to get logits for the next token
        float* logits = forward(&transformer, token, pos);

        if (pos < start_pos + num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next
            // prompt token
            next = prompt_tokens[pos - start_pos + 1];
        } else if (pos == end_pos - 2) {
            // reaching the end, force it to close by <|im_end|>
            next = 2;  // := <|im_end|>
        } else {
            // otherwise sample the next token from the logits
            next = sample(&sampler, logits);
        }

        pos++;

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(&tokenizer, token, next);
        if (pos >= num_prompt_tokens) {
            safe_printf(piece);  // same as printf("%s", piece), but skips
                                 // "unsafe" bytes
            fflush(stdout);
        }

        token = next;

        // init the timer here because the first iteration can be slower
        if (start_time == 0) {
            start_time = time_in_ms();
        }
    }
    printf("\n");

    long end_time = time_in_ms();
    // \033[0;32m set color to green and \033[0m reset to default, they won't be
    // generate by LLM
    fprintf(stdout, "\033[0;32mlength: %d, speed (tok/s): %.4f \033[0m\n", pos,
            (pos - start_pos) / (float)(end_time - start_time) * 1000);

    free(prompt_tokens);
}

int main(int argc, char* argv[]) {
    // default parameters
    char* model_path = "model.bin";  // e.g. out/model.bin
    char* tokenizer_path = "tokenizer.bin";
    float temperature =
        0.6f;  // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;  // top-p in nucleus sampling. 1.0 = off. 0.9 works well,
                        // but slower
    char* prompt = NULL;    // prompt strings
    int num_prompt = 0;     // number of prompts
    uint64_t rng_seed = 0;  // seed rng with time by default
    int num_thr = 0;

    if (argc == 4) {
        num_thr = atoi(argv[1]);
        rng_seed = atoi(argv[2]);
        prompt = argv[3];
    } else {
        fprintf(stderr, "Usage:   ./parallel <num_thr> <seed> <prompt>\n");
        fprintf(stderr,
                "Example: ./parallel 4 42 \"What is Fibonacci Number?\"\n");
        fprintf(stderr,
                "Note:    <prompt> must be quoted with \"\", only one prompt "
                "supported\n");
        exit(1);
    }

    // parameter validation/overrides
    if (num_thr <= 0 || num_thr > 16) {
        fprintf(stderr, "num_thr must between 1 and 16 \n");
        exit(EXIT_FAILURE);
    }
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);

    // build the Transformer via the model .bin file
    build_transformer(&transformer, model_path);
    // build the Tokenizer via the tokenizer .bin file
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    // build the Sampler
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp,
                  rng_seed);

    // initialize thread pool
    init_thr_pool(num_thr);

    printf("user: %s \n", prompt);
    // perform multi-threading generation
    generate(prompt);

    // close thread pool
    close_thr_pool();

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}