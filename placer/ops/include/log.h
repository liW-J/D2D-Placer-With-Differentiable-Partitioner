/*
 * @Date: 2025-03-15 21:37:46
 * @LastEditTime: 2025-04-08 21:15:53
 * @FilePath: /D2D-placer/placer/ops/include/log.h
 * @Description:
 */
#ifndef _LOG_H_
#define _LOG_H_

#include <stdio.h>
// #include "debug.h"

#ifdef __cplusplus 
extern "C" {
#endif

enum LogLevel
{
    ERROR = 1,
    WARN  = 2,
    INFO  = 3,
    DEBUG = 4,
};

void init_log(const char *log_file);   

void mylog(const char* filename, int line, enum LogLevel level, const char* fmt, ...) __attribute__((format(printf,4,5)));

#define LOG(level, format, ...) mylog(__FILE__, __LINE__, level, format, ## __VA_ARGS__)

#ifdef __cplusplus 
};
#endif

#endif