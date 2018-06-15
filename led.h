#pragma once
#include <vector>
#include <initializer_list>
#include <wiringPi.h>
#include <softPwm.h>
#include <stdexcept>

class led
{
private:
    std::vector<int> pins;
    std::vector<float> value;
    int range;
    bool clip;
public:
    led(std::vector<int>pins, std::vector<float> value, int range, bool clip)
        : pins(pins), value(value), range(range), clip(clip)
    {
        if(pins.size() != value.size())
            throw std::runtime_error("pins and value size missmatch");
        for(auto pin : pins)
        {
            if(softPwmCreate(pin, 0, range) != 0)
                throw std::runtime_error("Could not create software pwm");
        }
    }

    led(std::vector<int>pins, std::vector<float> value, int range)
        : led(pins, value, range, true)
    {
        
    }

    ~led();

    int channels()
    {
        return pins.size();
    }

    void setValue(std::vector<float> v)
    {
        if(v.size() != value.size())
            throw std::runtime_error("pins and value size missmatch");
        
        for(int i = 0; i < v.size(); i++)
        {
            float cur = v[i];
            if((cur <= 0 || cur >= 1) && !clip)
                throw std::runtime_error("Values must be between 1 and 0");
            else
            {
                cur = std::max(cur, 0.0f);
                cur = std::min(cur, 1.0f);
            }
            softPwmWrite(pins[i],(int)(range*cur));
            value[i] = cur;
        }
    }
};