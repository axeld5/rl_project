# Agent description

**Actions** : int: 0, 1, 2 = Left, Straight, Right

**State** : Array of size 11 :

- Distance to Danger Left : Bool (0 if the point next to head is collision)
- Distance to Danger Straight : Bool
- Distance to Danger Right : Bool
- Last move direction is Left : Bool
- Last move direction is Right : Bool
- Last move direction is Up : Bool
- Last move direction is Down : Bool
- Food is left : Bool
- Food is right : Bool
- Food is up : Bool
- Food is down : Bool

**Rewards** :   
- Eat food : +50
- Crash wall or eat his tail = Collision : -100
- Else : If the snake is closer to the food than last step : +1 ; if the snake is not closer to the food than last step : -2

**Collision** : If the screen size is (10, 15), then there is a collision when the snake moves to a point where $x = 0$ or $x = 10$ or $y = 0$ or $y = 15$.

NB : Axis are (x,y), the 0 is at the top left corner of the screen ; y point down (=> going "UP" on the screen means y _decreases_), x points right --> it's easier for the print.