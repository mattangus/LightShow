#include <stdio.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <wiringPi.h>
#include <softPwm.h>
#include <chrono>

#define RED 7
#define GREEN 9
#define BLUE 8

//gpioPulse_t pulse[2]; /* only need two pulses for a square wave */

class Colour
{
public:
  Colour(float r, float g, float b) : r(r), g(g), b(b) {}
  float r, g, b;

  Colour operator-(Colour o)
  {
    return Colour(r-o.r,g-o.g,b-o.b);
  }
  Colour operator+(Colour o)
  {
    return Colour(r+o.r,g+o.g,b+o.b);
  }
  Colour operator/(Colour o)
  {
    return Colour(r/o.r,b/o.b,g/o.g);
  }
  Colour operator/(float o)
  {
    return Colour(r/o,b/o,g/o);
  }
  Colour operator*(Colour o)
  {
    return Colour(r*o.r,b*o.b,g*o.g);
  }
  Colour operator*(float o)
  {
    return Colour(r*o,b*o,g*o);
  }
};

std::ostream& operator<<(std::ostream& str, Colour c)
{
  return str << "(" << c.r << ", " << c.g << ", " << c.b << ")";
}

void setColour(Colour c)
{
  Colour temp = c*100;
  softPwmWrite(RED,temp.r);
  softPwmWrite(GREEN,temp.g);
  softPwmWrite(BLUE,temp.b);
}

void fade(Colour from, Colour to, int ms)
{
  Colour step = (to-from)*10.0f/(float)ms;
  Colour temp = from;
  auto start = std::chrono::steady_clock::now();
  for(int i = 0;i < ms/10;i++)
  {
    setColour(temp);
    temp = temp + step;
    delay(10);
  }
}

int main(int argc, char *argv[])
{
  wiringPiSetup();
  if(softPwmCreate(RED, 0, 100) != 0)
    std::cout << "Something went wrong" << std::endl;
  if(softPwmCreate(GREEN, 0, 100) != 0)
    std::cout << "Something went wrong" << std::endl;
  if(softPwmCreate(BLUE, 0, 100) != 0)
    std::cout << "Something went wrong" << std::endl;

  Colour colours[] = {Colour(0,0,1), Colour(0,1,0), Colour(1,0,0)};

  for(int i = 0; i < 100; i++)
  {
    //std::cout << "fading from " << colours[i%3] << " to " << colours[(i+1)%3] << std::endl;
    fade(colours[i%3],colours[(i+1)%3], 5000);
  }
  return 0 ;
}
