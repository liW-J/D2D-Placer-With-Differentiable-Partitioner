/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-04-17 13:26:50
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-06-14 02:58:20
 * @FilePath: /D2D-placer/placer/ops/utils_3d/src/log.cpp
 * @Description: 
 */
/***************************************************************************************
* Copyright (c) 2014-2022 Zihao Yu, Nanjing University
*
* NEMU is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
*
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
*
* See the Mulan PSL v2 for more details.
***************************************************************************************/

/**
日志打印示例。
使用：
Log(DEBUG, "This is debug info\n");
结果：
[2018-07-22 23:37:27:172] [DEBUG] [main.cpp:5] This is debug info
默认打印当前时间（精确到毫秒）、文件名称、行号。
*/
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "include/common.h"

#ifndef LOGLEVEL
#define LOGLEVEL DEBUG
#endif

static void get_timestamp(char *buffer)
{
    time_t t;
    struct tm *p;
    struct timeval tv;
    int len;
    int millsec;

    t = time(NULL);
    p = localtime(&t);

    gettimeofday(&tv, NULL);
    millsec = (int)(tv.tv_usec / 1000);

    /* 时间格式：[2011-11-15 12:47:34:888] */
    len = snprintf(buffer, 32, "[%04d-%02d-%02d %02d:%02d:%02d:%03d] ",
        p->tm_year+1900, p->tm_mon+1,
        p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec, millsec);

    buffer[len] = '\0';
}

void mylog(const char* filename, int line, enum LogLevel level, const char* fmt, ...)
{
    if(level > LOGLEVEL)
        return;

    va_list arg_list;
    char buf[1024];
    memset(buf, 0, 1024);
    va_start(arg_list, fmt);
    vsnprintf(buf, 1024, fmt, arg_list);
    char time[32] = {0};

    // 去掉*可能*存在的目录路径，只保留文件名
    const char* tmp = strrchr(filename, '/');
    if (!tmp) tmp = filename;
    else tmp++;
    get_timestamp(time);

	switch(level){
		case DEBUG:
			//green
			printf("\033[1;32m%s[%s] [%s:%d] %s\033[0m \n", time, "DEBUG", tmp, line, buf);
      break;
		case INFO:
			//blue
			printf("\033[1;34m%s[%s] [%s:%d] %s\033[0m \n", time, "INFO", tmp, line, buf);
      break;
		case ERROR:
			//red
			printf("\033[1;31m%s[%s] [%s:%d] %s\033[0m \n", time, "ERROR", tmp, line, buf);
      break;
		case WARN:
			//yellow
			printf("\033[1;33m%s[%s] [%s:%d] %s\033[0m \n", time, "WARN", tmp, line, buf);
      break;
	}
    va_end(arg_list);
}

bool log_enable() {
  return true;
}