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
        int rows = initial.size();
        if(rows == 0)
            throw std::runtime_exception("gridState must have at least one row");

        int cols = initial[0].size();
        if(cols == 0)
            throw std::runtime_exception("gridState must have at least one column");
            
        int depth = initial[0][0].size();
        if(depth == 0)
            throw std::runtime_exception("gridState must have at least one channel");

        state = Eigen::Tensor<float, 3>(rows, cols, depth);

        for(int i = 0; i < initial.size(); i++)
        {
            for(int j = 0; j < initial[i].size(); j++)
            {
                for(int k = 0; k < initial[i][j].size(); k++)
                {
                    state(i,j,k) = initial[i][j][k];
                }
            }
        }
    }

    gridState lerp(gridState& other, float q)
    {
        return gridState(state*q + (1-q)*other.state);
    }

    ~gridState();
};
