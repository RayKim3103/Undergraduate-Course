#include "kernel/types.h"
#include "kernel/fcntl.h"
#include "user/user.h"

#define buf_size    128     // Max length of user input
#define max_args    16      // Max number of arguments

/***************************************************/
// Define strtok function
char *strtok(char *str, const char *delim);
/***************************************************/

int runcmd(char *cmd);      // Run a command.

// Read a shell input.
char* readcmd(char *buf) {
    // Read an input from stdin.
    fprintf(1, "$ ");
    memset(buf, 0, buf_size);
    char *cmd = gets(buf, buf_size);
  
    // Chop off the trailing '\n'.
    if(cmd) { cmd[strlen(cmd)-1] = 0; }
  
    return cmd;
}

int main(int argc, char **argv) {
    int fd = 0;
    char *cmd = 0;
    char buf[buf_size];
  
    // Ensure three file descriptors are open.
    while((fd = open("console", O_RDWR)) >= 0) {
        if(fd >= 3) { close(fd); break; }
    }
  
    fprintf(1, "EEE3535 Operating Systems: starting ysh\n");
  
    // Read and run input commands.
    while((cmd = readcmd(buf)) && runcmd(cmd)) ;
  
    fprintf(1, "EEE3535 Operating Systems: closing ysh\n");
    exit(0);
}

// Run a command.
int runcmd(char *cmd) {
    if(!*cmd) { return 1; }                     // Empty command

    // Skip leading white space(s).
    while(*cmd == ' ') { cmd++; }
    // Remove trailing white space(s).
    for(char *c = cmd+strlen(cmd)-1; *c == ' '; c--) { *c = 0; }

    if(!strcmp(cmd, "exit")) { return 0; }      // exit command
    else if(!strncmp(cmd, "cd ", 3)) {          // cd command
        if(chdir(cmd+3) < 0) { fprintf(2, "Cannot cd %s\n", cmd+3); }
    }
    else {
        // EEE3535 Operating Systems
        // Assignment 1: Shell
        /****************************************************************************************************/
        char *argv[max_args];      // argv array
        int background = 0;        // whether the program is in background(&) or not, background flag
        int i = 0;                 // index of argv

//        fprintf(2, "    EXEC: %s\n    Process id: %d, BG is: %d\n", cmd, getpid(), background);
        
        /* Variable that indicates the posiotion of series, pipe and redirection symbols */
        char *ser_pos = strchr(cmd, ';');
        char *pipe_pos = strchr(cmd, '|');
        char *redir_pos = strchr(cmd, '>');

/*************************************************************************************
How to handle series, background, pipe and redirection
1. First, if there is & in the end of cmd, set background flag to 1
- Now we should handle series, background, pipe, redirection
- I set the priority wit series being the highest, followd by pipe, and redirection
2. If there is ';', I seperated the cmd before and after ';' and execute runcmd
3. If there is '|', I seperated the cmd before and after '|' and execute runcmd

    - The behavior of the pipe was implemented as follows -

    First, p[2] is created, and the pipe’s writeEnd and readEnd are connected. 
    In the parent process, fork is called so that execute cmd1 writes to writeEnd, 
    and then another fork is called so that execute cmd2 reads from readEnd.

    The reason for using exit(0) is that when executing a single cmd, fork is called to execute single cmd in a child process.
    Once the execution is finished, it returns to the parent process, which is actually the child process created for handling the pipe. 
    To return to the original parent process where the pipe handling started, this child process needs to call exit(0).

    If there are three pipe commands (e.g., cmd1 | cmd2 | cmd3), fork is called in the parent process so that cmd1 writes to writeEnd, 
    then another fork is called so that cmd2 | cmd3 reads what is written to the pipe. 
    After that, cmd2 | cmd3 is handled in the same way as two pipe commands.

4. If there is '>', I seperated the cmd before '>' and execute runcmd and the output is written of righcmd which is file name
- I used, close(1) and dup(fd) to redirect the output of leftcmd to file
**************************************************************************************/

        // when '&' is on the last part of cmd
        if (cmd[strlen(cmd) - 1] == '&') 
        {
//            fprintf(2, "    BG became 1, PROC_ID:%d\n", getpid());
            cmd[strlen(cmd) - 1] = 0; // erase &
            background = 1;           // set background flag to 1
        }

        // handling series(';')
        if (ser_pos) 
        {
            *ser_pos = '\0'; // seperate left, right cmd using '\0'

            // in case of exit is executed in the middle. (example) whoami; exit; stressfs
            // if exit is executed, the ysh terminates, so the subsequnt commands will not executed
            if(!strcmp(cmd, "exit")) 
            {
                return runcmd(cmd);
            }

            runcmd(cmd); // execute the left cmd, wait(0)is called when cmd executes, so the runcmd(ser_pos + 1) can wait
            return runcmd(ser_pos + 1); // execute the right cmd, if "exit" it returns 0
        }

        // handling pipe('|')
        else if (pipe_pos) 
        {
            *pipe_pos = '\0'; // seperate left, right cmd using '\0'
            int p[2];  // generate filedescriptor
            pipe(p);   // pipe to readend and writeend has made

            if (fork() == 0) 
            {
                // execute leftcmd, in child process -> write pipe
                close(1);        // close stdout
                dup(p[1]);       // connect writeend of pipe to stdout
                close(p[0]);     // close pipe readend filedescriptor, this connection is not used so I closed it
                close(p[1]);     // close pipe writeend filedescriptor, this connection is not used so I closed it
                runcmd(cmd);     // execute leftcmd
                exit(0);         // exit process, I wrote why should exit on the above
            }
            if (fork() == 0) 
            {
                // execute leftcmd, in child process -> read pipe
                close(0);        // close stdin
                dup(p[0]);       // connect readend of pipe to stdin
                close(p[0]);     // close pipe readend filedescriptor, this connection is not used so I closed it
                close(p[1]);     // close pipe writeend filedescriptor, this connection is not used so I closed it
                runcmd(pipe_pos + 1); // execute rightcmd
                exit(0);         // exit process, I wrote why should exit on the above
            }

            // close pipe and wait for child process, if the process is not in background 
            close(p[0]);
            close(p[1]);
            if (!background) 
            {
                wait(0);
                wait(0);
            }
            else
            {
                background = 0; // Initiate background to 0, e.g. cmd1 | cmd2 | cmd3 & is executed on background but cmd1 | cmd2 | cmd3 itself should be executed in order
            }
            // return 1;
        }

        // handling redirection('>')
        else if (redir_pos) 
        {
            *redir_pos = '\0';  // seperate left, right cmd using '\0'
            char *file = strtok(redir_pos + 1, " "); // get file name using strtok, which will be same as rightcmd
        
            if (fork() == 0) 
            {
                // handling redirection in child process
                int fd = open(file, O_WRONLY | O_CREATE | O_TRUNC); // make or open file with O_WRONLY|O_CREATE|O_TRUNC flags, filename = file 
                if (fd < 0) 
                {
                    fprintf(2, "fail to open file");
                    exit(1);
                }
                
                close(1);      // close stdout
                dup(fd);       // connect filedescriptor to stdout
                close(fd);     // close filedesctiptor, original connection to filedescriptor is not used
                
                runcmd(cmd);   // runcmd
                exit(0);       // the reason to exit is same as handling pipe
            }
            // wait for child process, if the process is not in background 
            if (!background) 
            {
                wait(0);
            }
            else
            {
//                fprintf(2, "    BG back to 0, PROC_ID:%d\n", getpid());
                background = 0; // Initiate background to 0, e.g. cmd1 > cmd2 | ~~~ & is executed on background but cmd1 > cmd2 | ~~~ itself should be executed in order
            }
            // return 1;
        }

        // Executing single command
        else
        {
            // parse commands to handle many argv, e.g. ls cat echo init kill
            argv[i] = strtok(cmd, " ");
            while (argv[i] != 0 && i < max_args-1) 
            {
                argv[++i] = strtok(0, " ");
            }
            argv[i] = 0;

            // execute cmd using fork(), cmd execution is handled in child process
            if (fork() == 0) 
            {
                exec(argv[0], argv);   // execute cmd in child process
                fprintf(2, "exec %s failed\n", argv[0]); // if execution fails this printf will be printed
                exit(1);
            }

            // wait for child process, if the process is not in background 
            if (!background) 
            {
                wait(0);
            }
            else
            {
                background = 0; // Initiate background to 0
            }
        }
        /****************************************************************************************************/
    }
    return 1;
}

////////////////////////////////////////////////////////////////////////////////////////
/* The strtok function was not existed in xv6 */
/* Define strtok function */
char *strtok(char *str, const char *delim) {
    static char *next = 0; // static pointer which stores the next start position of string
    if (str != 0) 
    {
        next = str; // set next as the beginning of the string on first call
    }

    if (next == 0) 
    {
        return 0; // if there is no more string, return 0
    }

    // the start position of string
    char *start = next;

    // move the pointer until it meets delimiter
    while (*next != '\0') {
        const char *d = delim;
        while (*d != '\0') 
        {
            // when we found delimiter
            if (*next == *d) 
            { 
                *next = '\0'; // set end of the token to NULL
                next++;       // start from this position in next call -> this position: the position of delimiter + 1

                // To handle continuous delimiter
                if (start != next - 1) 
                {
                    return start;
                }

                start = next; // set new start position of string
                break;
            }
            d++;
        }
        next++;
    }

    // return the last token of string
    if (*start == '\0') 
    {
        return 0;
    } 
    else 
    {
        next = 0; // to return 0 at next call
        return start;
    }
}
//////////////////////////////////////////////////////////////////////////////////////

