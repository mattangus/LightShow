#pragma once
#include <vector>
#include "led.h"
#include "gridState.h"

class ledGrid
{
private:
    /**
     * @brief List of pins used for multiplexing the rows
     * 
     */
    std::vector<int> muxPins;
    /**
     * @brief List of LEDs (columns of grid)
     * 
     */
    std::vector<led> ledCol;
    /**
     * @brief Number of rows
     * This is needed as there may be less than 2^(muxPins.size()) rows
     * (i.e. not using all pins from multiplexer)
     */
    int rows;

    gridState currentState;
public:
    ledGrid(std::vector<int> muxPins, std::vector<led> colPins, int rows, int range)
        : muxPins(muxPins), ledCol(colPins), rows(rows)
    {
        
    }
    ~ledGrid();
};