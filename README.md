# TrafficControl
2022-2023 Capstone Engineering Project. 

## Table of Contents
* [Scheduled work](#documentation)
* [Code](#Code)



### Task:
Use cumulative knowledge to solve an important issue.



# Planning
<ln>
  
## Outline of project
*  Research the traffic patterns and issues around Charlottesville High School.
    * Use various data collection methods
      * drone footage
      * road-bump sensors
      * school climate surverys
    * Use data to create accurate traffic simulation
    * Use simulation to investigate possible solutions


## Materials
* Project is  simulation and computer-driven.
  * Visual Studio Code
  * Github
  * Python
* Must have a mechatronics aspect
  * Pressure plates on road detecting cars coming in and out
  * Possibly have car-detection software running on drone cameras


## Risk Mitigation
* Our project is pretty low risk
* Keeping drone at safe height, no higher than 120m
* No lower than 60m

# To-Do
* Gather data in morning and afternoon
  * Use drone to collect data
  * 
## Tentative schedule
* Yes

# Climate Survey results
* Yes



# Code
* There is so much code.
* I will explain the code when i understand the code.
* However, we did get some facial recognition and vehicle recognition working.
<img src="Media/millerni.png" alt="alt text" width="600" height="500">
<br/>
look! little squares!


# Traffic simulation plan
* Yes


# Documentation

### Week 6: 9/25/22
* Looked at last Friday's afternoon traffic video, tried to apply computer vision, but ran in to some problems. Also, this video starts at 3:59, which is after many people have been dismissed and already left the building
* Took drone video of morning traffic flow
* One issue that we are running in to is deciding which simulation software to use and learning how to use it. The programs that we have found so far have either not had the right traffic tools (Anylogic) or been very opaque and hard to learn (SUMO)
* Currently researching VISSIM, which could be a better balance of ease of use and applicability

### Week 7: 10/3/22
* C:/Users/Appdata/Local/Programs/Python/Python310/Lib/site-packages/labelImg
* https://pypi.org/project/labelImg/
* important lonks^
* Worked on finding simulation softwares. Went through multiple different ones, until we found one that accurately represents data, is easy to use, and does not have a time limit on simulation time.
* Worked on Computer vision Software, explored LabelImg, Anaconda, lxml, virtual environments, and other labeling stuff. 
* LabelImg program does not work in Python 3.10, had to fix float and int
* Learned a lot about training a model for computer vision
* Discussed a lot of potential solutions to simulation problems
* Next week will hopefully have a working model completed by Friday. 



### Week 8: 10/10/22
* https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
* this hurts
* predefined path class file - "C:\Users\wkeenan14\Documents\CHSTrafficCapstone\predefined_classes.txt"
* image_path -
"C:\Users\wkeenan14\Documents\CHSTrafficCapstone\data\images\train"
* Figured out how to define right of way at each intersection in VISSIM using conflict areas and priority rules
* Learned how to create pedestrian areas and routes
* Had trouble making pedestrians interact with vehicles
* Next week I will try to model the pedestrian-vehicle interactions and learn how to measure traffic flow in the simulation

### Week 9: 10/17/22
* Realized that my edits to the traffic simulation hadn't been saving since I was using the demo version, which doesn't allow saving or making changes to files, instead of the student version
* Took video of afternoon traffic conditions at the parent pick up line
* We realized that the scope of our project is far outside what is possible given the amount of time and power we have; the traffic flows well enough inside the school's campus, only being bottlenecked by the two intersections that cars can leave from. A roundabout could maybe improve this, but making a change that drastic wouldn't be reasonable to expect as the outcome of our project.
* Next week we will hopefuly come up with and flesh out an aidea for a new project
