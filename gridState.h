#pragma once
#include <stdexcept>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

typedef std::vector<std::vector<std::vector<float>>> primitiveGrid;

class gridState
{
private:
    Eigen::Tensor<float, 3> state;
public:

    gridState(Eigen::Tensor<float, 3> other) : state(other) { }

    gridState(primitiveGrid initial)
    {
        setGrid(initial);
    }

    gridState(int rows, int cols, int depth, float fill) : state(rows, cols, depth)
    {
        state.setConstant(fill);
    }

    void setGrid(primitiveGrid grid)
    {
        int rows = grid.size();
        if(rows == 0)
            throw std::runtime_error("gridState must have at least one row");

        int cols = grid[0].size();
        if(cols == 0)
            throw std::runtime_error("gridState must have at least one column");
            
        int depth = grid[0][0].size();
        if(depth == 0)
            throw std::runtime_error("gridState must have at least one channel");

        state = Eigen::Tensor<float, 3>(rows, cols, depth);

        for(int i = 0; i < grid.size(); i++)
        {
            for(int j = 0; j < grid[i].size(); j++)
            {
                for(int k = 0; k < grid[i][j].size(); k++)
                {
                    state(i,j,k) = grid[i][j][k];
                }
            }
        }
    }

    // gridState lerp(gridState& other, float q)
    // {
    //     return gridState(state*q + (1-q)*other.state);
    // }

    // void fade(Colour from, Colour to, int ms)
    // {
    //     Colour step = (to-from)*10.0f/(float)ms;
    //     Colour temp = from;
    //     auto start = std::chrono::steady_clock::now();
    //     for(int i = 0;i < ms/10;i++)
    //     {
    //         setColour(temp);
    //         temp = temp + step;
    //         delay(10);
    //     }
    // }

    ~gridState();
};
