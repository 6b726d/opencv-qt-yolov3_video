#ifndef TIMER_H
#define TIMER_H

#include <chrono>

struct timer {
    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::chrono::time_point<std::chrono::system_clock> end_time;
    bool running = false;

    void start() {
        start_time = std::chrono::system_clock::now();
        running = true;
    }

    void stop() {
        end_time = std::chrono::system_clock::now();
        running = false;
    }

    double elapsed_seconds() {
        std::chrono::time_point<std::chrono::system_clock> end_time_t;

        if (running) {
            end_time_t = std::chrono::system_clock::now();
        }
        else {
            end_time_t = end_time;
        }

        return std::chrono::duration_cast<std::chrono::seconds>(end_time_t - start_time).count();
    }
};

#endif // TIMER_H
