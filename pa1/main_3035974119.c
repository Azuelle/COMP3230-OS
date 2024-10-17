/*
 * PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
 * FILE NAME:
 * NAME:
 * UID:
 * Development Platform:
 * Remark: (How much you implemented?)
 * How to compile separately: (gcc -o main main_[UID].c)
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

// Define Global Variable, Additional Header, and Functions Here

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

    // Write your main logic here
    int pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // child process
        char* args[] = {"./inference", seed, NULL};
        execvp(args[0], args);
        perror("execvp");
        exit(EXIT_FAILURE);
    }
    // parent process
    dup2(STDIN_FILENO, STDIN_FILENO);

    return EXIT_SUCCESS;
}