#include "display.h"
#include "temp.h"
#include <stdio.h>
#include <stdlib.h>
#include <Servo.h>
  // create servo object to control a servo
// twelve servo objects can be created on most boards

int pos = 0;    // variable to store the servo position

extern const font_type TimesNewRoman;

  
Servo myservo;

  
class Robot {
  public: 

bool Lidar = true;
bool Cam = true;
bool rst = true;
void Display(uint32_t x, uint32_t y, char *str,uint16_t bgcolor=0x0000,uint16_t fontcolor=0xFFFF)
{
   LCD_PrintString(x,y,str,bgcolor,fontcolor); 
}
char* GetTemp(Adafruit_MLX90614 *mlx)
{
   uint32_t temp= mlx->readObjectTempC();
   char buff[4]; //СЂРµР·СѓР»СЊС‚Р°С‚
   char *p;  //СѓРєР°Р·Р°С‚РµР»СЊ РЅР° СЂРµР·СѓР»СЊС‚Р°С‚
   p = itoa(temp,buff,10);
   return p;
}  

void TurnServo()
{
  for (pos = 0; pos <= 180; pos += 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo.write(pos);              // tell servo to go to position in variable 'pos'
    delay(3);                       // waits 15ms for the servo to reach the position
                                      }
  for (pos = 180; pos >= 0; pos -= 1) { // goes from 180 degrees to 0 degrees
    myservo.write(pos);              // tell servo to go to position in variable 'pos'
    delay(3);                       // waits 15ms for the servo to reach the position
                                      }
  }


void ResetDisplay()
{
LCD_Fill(ILI9341_BLACK); 
}

void OnBoard()
{
}

char* VoltageToPercent()
{
  float voltage = 0.0;
  voltage = sensors.checkVoltage();
  char buff[6]; //СЂРµР·СѓР»СЊС‚Р°С‚
  char *p;  //СѓРєР°Р·Р°С‚РµР»СЊ РЅР° СЂРµР·СѓР»СЊС‚Р°С‚
  p = itoa(voltage,buff,10);
  return p;
}
  
};
