#include <stdio.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <wiringPi.h>
#include <softPwm.h>
#include <chrono>
#include <vector>

#include "ledGrid.h"
#include "led.h"

int main(int argc, char *argv[])
{
    std::vector<int> muxPins = {6, 10, 11};
    std::vector<std::vector<int>> ledPins = {{12, 13, 14},
                                             {0,  2,  3 },
                                             {8,  9,  7 },
                                             {15, 16, 1}};
    std::vector<led> columns;
    for(auto& pins : ledPins)
        columns.push_back(led(pins, {0.f, 0.f, 0.f}, 100));

    int depth = 3; //rgb

    ledGrid grid(muxPins, columns, 4);

    

    //for(int i = 0)
    //ledGrid grid({};
    return 0;
}
