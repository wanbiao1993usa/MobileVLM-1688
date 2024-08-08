#ifndef LOG_H
#define LOG_H
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#define DEBUG_DEMO


void LOG(){}

template<typename T, typename... Args>
void LOG(T first, Args... args) {
    #ifdef DEBUG_DEMO
    std::cout << first << " ";
    LOG(args...); // 递归调用
    #endif
}

#endif