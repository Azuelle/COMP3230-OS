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

#define DEBUG 0

int inference_pid = -1;   // pid of inference process
int inference_ready = 1;  // 0 if not ready, 1 if ready, set with SIGUSR1

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

int main(int argc, char* argv[]) {
    char* seed;  //
    if (argc == 2) {
        seed = argv[1];
    } else if (argc == 1) {
        // use 42, the answer to life the universe and everything, as default
        seed = "42";
    } else {
        fprintf(stderr, "Usage: ./main <seed>\n");
        fprintf(stderr, "Note:  default seed is 42\n");
        exit(1);
    }

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

    // Main loop
    while (1) {
        // Wait for response to finish
        if (!inference_ready) {
            sched_yield();
            continue;
        }
        // READY
        !DEBUG ?: puts("[MAIN] GOT INF READY");
        inference_ready = 0;
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
        kill(inference_pid, SIGUSR1);  // Notify inference process
        !DEBUG ?: puts("[MAIN] SENT PROMPT");
    }

    // Clean up
    if (fclose(pipe_w) == EOF) {
        perror("fclose failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }
    if (close(pipefd[WRITE_END]) == -1) {
        perror("close failed");
        kill(inference_pid, SIGINT);
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}