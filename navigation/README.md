# Quick Start: Navigation Tutorials


## Move 1 metre along X axis
```bash
python navigate.py --dx 1
```

## Move 5 metres along X axis, square obstacles(1m X 1m) centred around (2.5,0)
- Make sure that a square obstacle (size 1m X 1m) is placed at (2.5,0)
```commandline
python navigate.py --dx 5
```

## Move 5 metres along X axis, square obstacles(1m X 1m) marked as "no-go" areas
- This script requires an obstacle map which is a JSON file containing (x,y) locations for obstacles along with the waypaths. 
```commandline
python navigate_around_nogo.py ROBOT_IP obstacle_map1.json 
```

## Start at (0,0), move to (5, 0). 
    - Square obstacle(1m X 1m) placed at (2.5, 0)
    - Move From (5, 0) to (10, 5)
        - Square obstacle (1m X 1m) placed at (7.5, 2.5)
    - Move from (10, 5) to (10, 10)
```commandline
python navigate_around_nogo.py ROBOT_IP obstacle_map2.json 
```
