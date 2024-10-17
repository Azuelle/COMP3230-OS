/*
 * PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
 * FILE NAME: main_3035974119.c
 * NAME: TANG Jiakai
 * UID: 3035974119
 * Development Platform: Ubuntu 22.04.4 LTS aarch64
 *                       (in Docker container on macOS)
 * Remark: (How much you implemented?)
 * How to compile separately: gcc -o main main_3035974119.c
 */

#include "common.h"  // common definitions

#include <stdio.h>    // for printf, fgets, scanf, perror
#include <stdlib.h>   // for exit() related
#include <unistd.h>   // for folk, exec...
#include <wait.h>     // for waitpid
#include <signal.h>   // for signal handlers and kill
#include <string.h>   // for string related
#include <sched.h>    // for sched-related
#include <syscall.h>  // for syscall interface

#define READ_END 0      // helper macro to make pipe end clear
#define WRITE_END 1     // helper macro to make pipe end clear
#define SYSCALL_FLAG 0  // flags used in syscall, set it to default 0

#include <errno.h>  // for reading errno

#define DEBUG 0

int inference_pid = -1;   // pid of inference process
int inference_ready = 0;  // set with SIGUSR1
int finished = 0;         // set with SIGUSR2

// Kill inference process when SIGINT is received
void handle_sigint(int signum) {
    kill(inference_pid, SIGINT);
    int status;
    waitpid(inference_pid, &status, SYSCALL_FLAG);
    printf("Child exited with %d\n", status);
    exit(EXIT_SUCCESS);
}

// Start new prompt input when SIGUSR1 is received
void handle_sigusr1(int signum) { inference_ready = 1; }

// End when SIGUSR2 is received
void handle_sigusr2(int signum) { finished = 1; }

// Set scheduling policy and nice value (if given)
void set_inference_policy(char* policy_str, int nice) {
    if (policy_str == NULL) return;

    // Get policy as int from name
    int policy = SCHED_OTHER;  // Default policy
    if (policy_str == "SCHED_BATCH")
        policy = SCHED_BATCH;
    else if (policy_str == "SCHED_IDLE")
        policy = SCHED_IDLE;
    // Realtime policies are ignored for now

    struct sched_attr attr = {.sched_policy = policy, .sched_nice = nice};
    if (syscall(SYS_sched_setattr, inference_pid, &attr, 0) == -1) {
        perror("syscall sched_setattr failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }
}

long unsigned last_utime = 0, last_stime = 0;

FILE* open_stat_file() {
    // Get stat file path
    char buffer[256];
    int ret = snprintf(buffer, sizeof(buffer), "/proc/%d/stat", inference_pid);
    if (ret >= sizeof(buffer)) {
        perror("inference stat path too long");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }
    if (ret < 0) {
        perror("snprintf failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }

    // Open stat file
    FILE* stat = fopen(buffer /*stat_path*/, "r");
    if (stat == NULL) {
        perror("fopen failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }

    return stat;
}

// Init last_utime and stime
void init_times() {
    FILE* stat = open_stat_file();

    // Read required fields
    if (fscanf(stat,
               "%*d %*s %*c %*d %*d "  // 5
               "%*d %*d %*d %*u %*u "  // 10
               "%*u %*u %*u %lu %lu "  // 15   // (14) utime, (15) stime
               ,
               &last_utime, &last_stime) <= 0) {
        perror("fscanf failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }

    if (fclose(stat) == EOF) {
        perror("fclose failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }
}

// Print inference process stats to stderr during generation
void print_inference_stats() {
    FILE* stat = open_stat_file();

    // Read required fields
    char tcomm[256], state;
    long unsigned utime, stime, vsize;
    long nice;
    unsigned policy;
    int task_cpu;
    if (fscanf(stat,
               "%*d %s %c %*d %*d "    // 5    // (2) tcomm, (3) state
               "%*d %*d %*d %*u %*u "  // 10
               "%*u %*u %*u %lu %lu "  // 15   // (14) utime, (15) stime
               "%*d %*d %*d %ld %*d "  // 20   // (19) nice
               "%*d %*u %lu %*d %*u "  // 25   // (23) vsize
               "%*u %*u %*u %*u %*u "  // 30
               "%*u %*u %*u %*u %*u "  // 35
               "%*u %*u %*d %d %*u "   // 40   // (39) task_cpu
               "%u %*u %*u %*d %*u "   // 45   // (41) policy
               ,
               tcomm, &state, &utime, &stime, &nice, &vsize, &task_cpu,
               &policy) <= 0) {
        perror("fscanf failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }

    if (fclose(stat) == EOF) {
        perror("fclose failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }

    // Print stats to stderr
    if (fprintf(
            stderr,
            "[pid] %d [tcomm] %s [state] %c [policy] %s [nice] %ld [vsize] %lu "
            "[task_cpu] %d [utime] %lu [stime] %lu [cpu%%] %.2f%%\n",
            inference_pid, tcomm, state, get_sched_name(policy), nice, vsize,
            task_cpu, utime, stime,
            ((utime - last_utime) + (stime - last_stime)) / 0.3) < 0) {
        perror("fprintf failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }

    // Update last_time values
    last_utime = utime;
    last_stime = stime;
}

void usage_err() {
    fprintf(stderr,
            "Usage: ./main [seed] or ./main <seed> <scheduling policies> "
            "[priority]\n");
    fprintf(stderr, "Note:  default seed is 42\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    char *seed, *policy = NULL;
    int nice = 0;
    if (argc > 4) usage_err();
    if (argc >= 2) {
        if (argc == 4) nice = atoi(argv[3]);
        if (argc >= 3) policy = argv[2];
        seed = argv[1];
    } else if (argc == 1) {
        // use 42, the answer to life the universe and everything, as default
        seed = "42";
    } else
        usage_err();

    // Create pipe
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        perror("pipe failed");
        exit(EXIT_FAILURE);
    }

    // Fork and exec inference process
    inference_pid = fork();
    if (inference_pid < 0) {
        perror("fork failed");
        exit(EXIT_FAILURE);
    } else if (inference_pid == 0) {
        // CHILD PROCESS

        // Redirect to stdin
        if (close(pipefd[WRITE_END]) == -1) {
            perror("close failed");
            exit(EXIT_FAILURE);
        }
        if (dup2(pipefd[READ_END], STDIN_FILENO) == -1) {
            perror("dup2 failed");
            exit(EXIT_FAILURE);
        }

        // Exec inference process
        char* args[] = {"./inference", seed, NULL};
        execvp(args[0], args);
        perror("execvp failed");
        exit(EXIT_FAILURE);
    }
    // PARENT PROCESS

    // Close unused pipe end
    if (close(pipefd[READ_END]) == -1) {
        perror("close failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }

    // Open write end of pipe
    FILE* pipe_w = fdopen(pipefd[WRITE_END], "w");
    if (pipe_w == NULL) {
        perror("fdopen failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }

    // Define signal handlers
    if (signal(SIGINT, handle_sigint) == SIG_ERR) {
        perror("signal SIGINT failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }
    if (signal(SIGUSR1, handle_sigusr1) == SIG_ERR) {
        perror("signal SIGUSR1 failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }
    if (signal(SIGUSR2, handle_sigusr2) == SIG_ERR) {
        perror("signal SIGUSR2 failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }

    set_inference_policy(policy, nice);

    // Main loop
    while (!finished) {
        // Read prompt from user
        char prompt[MAX_PROMPT_LEN];
        printf(">>> ");
        if (fgets(prompt, MAX_PROMPT_LEN, stdin) == NULL) {
            perror("fgets failed");
            kill(inference_pid, SIGINT);
            exit(EXIT_FAILURE);
        }

        // Send prompt to inference process
        if (fputs(prompt, pipe_w) == EOF) {
            perror("fputs failed");
            kill(inference_pid, SIGINT);
            exit(EXIT_FAILURE);
        }
        if (fflush(pipe_w) == EOF) {
            perror("fflush failed");
            kill(inference_pid, SIGINT);
            exit(EXIT_FAILURE);
        }
        !DEBUG ?: puts("[MAIN] SENT PROMPT");

        // Monitor stats while waiting for generation to finish
        init_times();
        while (!inference_ready && !finished) {
            if (usleep(300000) == -1 && errno != EINTR) {  // Sleep for 300ms
                perror("usleep failed");
                kill(inference_pid, SIGINT);
                exit(EXIT_FAILURE);
            }
            print_inference_stats();
        }
        // READY
        !DEBUG ?: puts("[MAIN] GOT INF READY");
        inference_ready = 0;
    }

    // Clean up
    if (fclose(pipe_w) == EOF) {
        perror("fclose failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
