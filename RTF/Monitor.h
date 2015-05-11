// File:   Monitor.h
// Author: t-jejan
//
// Implements functionality related to printing of debug/status information.
//
#ifndef _H_MONITOR_H_
#define _H_MONITOR_H_

#include <stdio.h>
#include <stdarg.h>
#include <string>
#include <time.h>

namespace Monitor
{
    // The default monitor that is used to print debug/status info to the console. In addition
    // to a user-provided message, it also prints details about CPU usage and memory consumption.
    class DefaultMonitor
    {
    public:
        static void Display(const char* fmt, ...)
        {
            va_list args;
            va_start(args, fmt);
            vfprintf(stderr, fmt, args);
            va_end(args);
            fflush(stderr);
        }
        static void Report(const char* fmt, ...)
        {
            va_list args;
            va_start(args, fmt);
            DefaultMonitor::ReportVA(fmt, args);
            va_end(args);
        }

        static void ReportVA(const char* fmt, va_list argptr)
        {
            #pragma omp critical
            {
                time_t now;
                time(&now);
                std::string timestr(ctime(&now));
                timestr.erase(timestr.end() - 1);

                fprintf(stderr, "-- %s --\n", timestr.c_str());

                vfprintf(stderr, fmt, argptr);
                fflush(stderr);
            }
        }
    };

    class NullMonitor
    {
    public:
        static void Display(const char* fmt, ...)
        {

        }
        static void Report(const char* fmt, ...)
        {

        }

        static void ReportVA(const char* fmt, va_list argptr)
        {

        }
    };
}

#endif // _H_MONITOR_H_
