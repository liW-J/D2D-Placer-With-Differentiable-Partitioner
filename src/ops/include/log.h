/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-03-15 21:37:46
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-03-15 21:45:05
 * @FilePath: /D2D-placer/src/ops/include/log.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef _LOG_H_
#define _LOG_H_

#include <stdio.h>
#include "debug.h"

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

void mylog(const char* filename, int line, enum LogLevel level, const char* fmt, ...) __attribute__((format(printf,4,5)));

#define LOG(level, format, ...) mylog(__FILE__, __LINE__, level, format, ## __VA_ARGS__)

#ifdef __cplusplus 
};
#endif

#endif