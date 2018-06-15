#pragma once
#include <vector>
#include <algorithm>
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

    static void checkCol(std::vector<led> ledCol)
    {
        if (ledCol.size() == 0)
            throw std::runtime_error("Led columns must have at least one element");
        
        int depth = ledCol[0].channels();
        for( auto l : ledCol)
        {
            if(depth != l.channels())
                throw std::runtime_error("Channel missmatch");
        }
    }

public:
    ledGrid(std::vector<int> muxPins, std::vector<led> colPins, int rows)
        : muxPins(muxPins), ledCol(colPins), rows(rows),
                currentState(rows, colPins.size(),
                (checkCol(ledCol), ledCol[0].channels()), 0.0f)
    {

    }

    

    ~ledGrid();
};