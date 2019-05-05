#include <stdio.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <pigpio.h>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <signal.h>

#define USE_PWM 0

class Colour
{
public:
    Colour(float r, float g, float b) : r(r), g(g), b(b) {}
    float r, g, b;

    Colour operator-(Colour o)
    {
        return Colour(r - o.r, g - o.g, b - o.b);
    }
    Colour operator+(Colour o)
    {
        return Colour(r + o.r, g + o.g, b + o.b);
    }
    Colour operator/(Colour o)
    {
        return Colour(r / o.r, g / o.g, b / o.b);
    }
    Colour operator/(float o)
    {
        return Colour(r / o, g / o, b / o);
    }
    Colour operator*(Colour o)
    {
        return Colour(r * o.r, g * o.g, b * o.b);
    }
    Colour operator*(float o)
    {
        return Colour(r * o, g * o, b * o);
    }
};

std::ostream &
operator<<(std::ostream &str, Colour c)
{
    return str << "(" << c.r << ", " << c.g << ", " << c.b << ")";
}

class LED
{
private:
    uint8_t r_pin, g_pin, b_pin;
public:
    Colour curColour;
    LED(uint8_t r_pin, uint8_t g_pin, uint8_t b_pin)
        : r_pin(r_pin), g_pin(g_pin), b_pin(b_pin), curColour(0,0,0)
    {
        gpioSetMode(r_pin, PI_OUTPUT);
        gpioSetMode(g_pin, PI_OUTPUT);
        gpioSetMode(b_pin, PI_OUTPUT);
    }

    ~LED()
    {
        gpioWrite(r_pin, 0);
        gpioWrite(g_pin, 0);
        gpioWrite(b_pin, 0);
    }

    void setColour(Colour& c)
    {
        setColour(c.r, c.g, c.b);
    }

    void setColour(float r, float g, float b)
    {
        curColour.r = r;
        curColour.g = g;
        curColour.b = b;

        // #if USE_PWM
        // Colour temp = curColour * 255;
        // gpioPWM(r_pin, temp.r);
        // gpioPWM(g_pin, temp.g);
        // gpioPWM(b_pin, temp.b);
        // #else
        gpioWrite(r_pin, r != 0);
        gpioWrite(g_pin, g != 0);
        gpioWrite(b_pin, b != 0);
        // #endif
    }

    // void makeWave(std::vector<Colour> colours, int hz)
    // {
    //     for(int i = 0; i < channels.size(); i++)
    //     {
    //         if(channels[i] >= 8)
    //         {
    //             std::stringstream ss;
    //             ss << channels[i] << " is an invalid channel for mux";
    //             throw std::runtime_error(ss.str());
    //         }
    //     }

    //     int n_pulse = channels.size() + 1;

    //     int cyc_len = (1e6 / n_pulse) / hz;

    //     gpioPulse_t pulses[n_pulse];

    //     int prevOn = 0;

    //     for(int i = 0; i < n_pulse; i++)
    //     {
    //         pulses[i].gpioOn =  ((channels[i] & 1) << a_pin) |
    //                             ((channels[i] & 2) << b_pin) |
    //                             ((channels[i] & 4) << c_pin);
    //         pulses[i].gpioOff = prevOn;
    //         pulses[i].usDelay = cyc_len;

    //         prevOn = pulses[i].gpioOn;
    //     }

    //     if(gpioWaveAddGeneric(n_pulse, pulses) == PI_TOO_MANY_PULSES)
    //     {
    //         throw std::runtime_error("too many pulses");
    //     }
    // }
};

std::ostream &
operator<<(std::ostream &str, LED l)
{
    return str << l.curColour;
}

class Mux
{
private:
public:
    int inhib_pin, a_pin, b_pin, c_pin;
    Mux(int inhib_pin, int a_pin, int b_pin, int c_pin)
        : inhib_pin(inhib_pin), a_pin(a_pin), b_pin(b_pin), c_pin(c_pin)
    {
        gpioSetMode(inhib_pin, PI_OUTPUT);
        gpioSetMode(a_pin, PI_OUTPUT);
        gpioSetMode(b_pin, PI_OUTPUT);
        gpioSetMode(c_pin, PI_OUTPUT);
    }

    ~Mux()
    {
        gpioWrite(a_pin, 0);
        gpioWrite(b_pin, 0);
        gpioWrite(c_pin, 0);
        gpioWrite(inhib_pin, 0);
    }

    void selectChannel(int channel)
    {
        if(channel >= 8)
        {
            std::stringstream ss;
            ss << channel << " is an invalid channel for mux";
            throw std::runtime_error(ss.str());
        }
        if(channel < 0)
        {
            off();
            return;
        }

        gpioWrite(a_pin, channel & 1);
        gpioWrite(b_pin, channel & 2);
        gpioWrite(c_pin, channel & 4);

        on();
    }

    void off()
    {
        gpioWrite(inhib_pin, 1);
    }

    void on()
    {
        gpioWrite(inhib_pin, 0);
    }

    void makeWave(std::vector<int> channels, int hz)
    {
        for(int i = 0; i < channels.size(); i++)
        {
            if(channels[i] >= 8)
            {
                std::stringstream ss;
                ss << channels[i] << " is an invalid channel for mux";
                throw std::runtime_error(ss.str());
            }
        }

        int n_pulse = channels.size() + 1;

        int cyc_len = (1e6 / n_pulse) / hz;

        gpioPulse_t pulses[n_pulse];

        int prevOn = 0;

        for(int i = 0; i < n_pulse; i++)
        {
            pulses[i].gpioOn =  ((channels[i] & 1) << a_pin) |
                                ((channels[i] & 2) << b_pin) |
                                ((channels[i] & 4) << c_pin);
            pulses[i].gpioOff = prevOn;
            pulses[i].usDelay = cyc_len;

            prevOn = pulses[i].gpioOn;
        }

        if(gpioWaveAddGeneric(n_pulse, pulses) == PI_TOO_MANY_PULSES)
        {
            throw std::runtime_error("too many pulses");
        }
    }
};

class Grid
{
    std::vector<std::vector<std::shared_ptr<Colour>>> _grid;
public:
    int numRows, numCols;
    Grid(int numRows, int numCols) : numRows(numRows), numCols(numCols)
    {
        for(int i = 0; i < numRows; i++)
        {
            std::vector<std::shared_ptr<Colour>> curRow;
            for(int j = 0; j < numCols; j++)
            {
                auto curColour = std::make_shared<Colour>(0,0,0);
                curRow.push_back(curColour);
            }
            _grid.push_back(curRow);
        }
    }

    std::shared_ptr<Colour> get(int row, int col)
    {
        if(!(row < numRows && col < numCols))
        {
            std::stringstream ss;
            ss << "Invalid index into grid [" << row << "," << col << "]";
            ss << " with shape [" << numRows << "," << numCols << "].";
            throw std::runtime_error(ss.str());
        }

        return _grid[row][col];
    }
};

std::ostream &
operator<<(std::ostream &str, Grid g)
{
    for(int i = 0; i < g.numRows; i++)
    {
        for(int j = 0; j < g.numCols; j++)
        {
            auto colour = g.get(i,j);
            str << *colour << ",\t";
        }
        str << std::endl;
    }
    return str;
}

class Display
{
private:
    std::shared_ptr<Grid> colours;
    std::shared_ptr<Mux> mux;
    std::vector<std::shared_ptr<LED>> leds; 
public:
    Display(/* args */) { }
    ~Display() { }
};

// void draw(uint8_t buffer[ROWS][COLS], uint8_t delay_ms)
// {

//     for (uint8_t row = 0; row < ROWS; row++)
//     {
//         /* Connect or disconnect columns as needed. */
//         for (uint8_t column = 0; column < COLS; column++)
//         {
//             digitalWrite(column_pins[column], buffer[row][column]);
//         }

//         /* Turn on whole row. */
//         digitalWrite(row_pins[row], LOW);

//         delay(delay_ms);

//         /* Turn off whole row. */
//         digitalWrite(row_pins[row], HIGH);
//     }
// }

// void setColour(Colour c)
// {
//     Colour temp = c * 100;
//     softPwmWrite(RED, temp.r);
//     softPwmWrite(GREEN, temp.g);
//     softPwmWrite(BLUE, temp.b);
// }

// void fade(Colour from, Colour to, int ms)
// {
//     Colour step = (to - from) * 10.0f / (float)ms;
//     Colour temp = from;
//     auto start = std::chrono::steady_clock::now();
//     for (int i = 0; i < ms / 10; i++)
//     {
//         setColour(temp);
//         temp = temp + step;
//         delay(10);
//     }
// }

// void init()
// {

//     /* Turn all columns off by setting then low. */
//     for (uint8_t x = 0; x < COLS; x++)
//     {
//         pinMode(column_pins[x], OUTPUT);
//         digitalWrite(column_pins[x], LOW);
//     }

//     /* Turn all rows off by setting then high. */
//     for (uint8_t y = 0; y < ROWS; y++)
//     {
//         pinMode(row_pins[y], OUTPUT);
//         digitalWrite(row_pins[y], HIGH);
//     }
// }

void mainLoop()
{
    LED l1(27,18,17);
    LED l2(24,23,22);

    Mux m(13,19,26,12);

    int channel = 0;

    Colour red(1,0,0);
    Colour green(0,1,0);
    Colour blue(0,0,1);
    Colour black(0,0,0);

    l1.setColour(blue);
    m.selectChannel(0);

    for(;;)
    {
        
        l1.setColour(black);
        l2.setColour(black);
        m.selectChannel(channel);
        if(channel == 0)
        {
            l1.setColour(red);
            l2.setColour(blue);
        }
        else
        {
            l1.setColour(green);
            l2.setColour(green);
        }

        gpioDelay(100);

        channel ^= 1;
    }
}

// void test()
// {
//     std::cout << "setting up outputs" << std::endl;
//     std::cout << "gpioSetMode: " << gpioSetMode(13, PI_OUTPUT) << std::endl;
//     std::cout << "gpioSetMode: " << gpioSetMode(19, PI_OUTPUT) << std::endl;
//     std::cout << "gpioSetMode: " << gpioSetMode(26, PI_OUTPUT) << std::endl;
//     std::cout << "gpioSetMode: " << gpioSetMode(12, PI_OUTPUT) << std::endl;

//     std::cout << "gpioSetMode: " << gpioSetMode(17, PI_OUTPUT) << std::endl;
//     std::cout << "gpioSetMode: " << gpioSetMode(18, PI_OUTPUT) << std::endl;
//     std::cout << "gpioSetMode: " << gpioSetMode(27, PI_OUTPUT) << std::endl;
//     std::cout << "gpioSetMode: " << gpioSetMode(22, PI_OUTPUT) << std::endl;
//     std::cout << "gpioSetMode: " << gpioSetMode(23, PI_OUTPUT) << std::endl;
//     std::cout << "gpioSetMode: " << gpioSetMode(24, PI_OUTPUT) << std::endl;

//     std::cout << "gpioWrite: " << gpioWrite(13, 0) << std::endl;
//     std::cout << "gpioWrite: " << gpioWrite(19, 0) << std::endl;
//     std::cout << "gpioWrite: " << gpioWrite(26, 0) << std::endl;
//     std::cout << "gpioWrite: " << gpioWrite(12, 0) << std::endl;

//     std::vector<int> pins = {17,18,27,22,23,24};

//     for(int i = 0; i < 6; i++)
//     {
//         int p = pins[i];
//         std::cout << "writing " << p << " 1" << std::endl;
//         std::cout << "gpioWrite: " << gpioWrite(p, 0) << std::endl;

//         std::cout << "gpioDelay: " << gpioDelay(1000000) << std::endl;
//     }
// }
void test()
{
    Mux m(13,19,26,12);

    LED l1(17,18,27);
    LED l2(22,23,24);

    l1.setColour(0,1,0);
    l2.setColour(0,1,1);
    
    gpioWaveClear();
    m.makeWave({0,1}, 100);
    // const int n_pulse = 2;
    // gpioPulse_t pulses[n_pulse];

    // pulses[0].gpioOn = (1 << 17);
    // pulses[0].gpioOff = 0;
    // pulses[0].usDelay = 1000;
    // // 1,010,000
    
    // std::cout << "on: " << pulses[0].gpioOn << std::endl;

    // pulses[1].gpioOn = 0;
    // pulses[1].gpioOff = (1 << 17);
    // pulses[1].usDelay = 1000;

    // // gpioWaveAddNew();

    // gpioWaveAddGeneric(n_pulse, pulses);

    int wave_id = gpioWaveCreate();

    if (wave_id >= 0)
    {
        std::cout << "wave micros " << gpioWaveGetMicros() << std::endl;
        gpioWaveTxSend(wave_id, PI_WAVE_MODE_REPEAT);

        std::cout << "wave started " << wave_id << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(10));

        gpioWaveTxStop();
    }
    else
    {
        std::cout << "didn't work" << std::endl;
    }
}

void sigint(int a)
{
    gpioTerminate();
    exit(0);
}

int main(int argc, char *argv[])
{
    signal(SIGINT, sigint);

    int status = gpioInitialise();
    if(status < 0)
    {
        std::cout << "Failed to init" << std::endl;
        exit(status);
    }

    try
    {
        // mainLoop();
        test();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    gpioTerminate();
    return 0;
}
