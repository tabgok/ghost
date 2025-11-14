// This example shows how to blink the three user LEDs on the
// 3pi+ 32U4.

#include <Pololu3piPlus32U4.h>

using namespace Pololu3piPlus32U4;

void setup()
{

}

void loop()
{
  // Turn the LEDs on.
  ledRed(1);
  ledYellow(0);
  ledGreen(0);
  delay(1000);

  ledRed(0);
  ledYellow(1);
  ledGreen(0);
  delay(1000);

  ledRed(0);
  ledYellow(0);
  ledGreen(1);
  delay(1000);
}