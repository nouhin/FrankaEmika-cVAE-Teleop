# Teleoperation with Interactive Learning Models

This project was developed during my Master's at ENSAM, Master SAR. Our aim was to simplify the teleoperation of the Franka Emika Robot by using cVAE, addressing the challenge of controlling its 7 joints.

For more details on the foundational research, please refer to the paper: [Controlling Assistive Robots with Learned Latent Actions by Dylan P. Losey, Krishnan Srinivasan, Ajay Mandlekar, Animesh Garg, Dorsa Sadigh](https://collab.me.vt.edu/pdfs/losey_icra2020.pdf).

## Requirements

The project was developed and tested on ROS Kinetic with Ubuntu 14.04, with the following requirements:

- Python 2.7
- PyTorch 1.4
- NumPy

## Installation

1. **ROS Kinetic on Ubuntu 14.04**: Follow the instructions [here](http://wiki.ros.org/kinetic/Installation/Ubuntu).
2. **Franka Emika ROS Libraries**:

```
sudo apt install ros-kinetic-libfranka ros-kinetic-franka-ros

```

3. **Joystick Support**:

```
sudo apt-get install ros-kinetic-joy

```

4. **Catkin Package**:

Navigate to the provided catkin package directory and run:

```
catkin_make -j4 -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH=/path/to/libfranka/build
source devel/setup.bash
```

## Training cVAE on HARMONIC Data

1. Ensure all scripts are at the same level as the "Dataset" folder containing the HARMONIC dataset.
2. Run `dataParser.py` to prepare the data.
3. Use `train.py` to train the cVAE defined in `model.py`.
4. Checkpoints are saved in a "models" folder.
5. To evaluate the trained model on new data, use `test.py` with the checkpoint file at the same level as the script. Ensure the model name is correctly specified.

## Manual Teleoperation

### Keyboard Control

- **Commands**: Use specific key combinations to control the robot's end-effector position and orientation.
- **Launch Instructions**: 
```
roslaunch panda_gazebo panda_rviz.launch
rosrun panda_gazebo keyboard_teleop.py
```

### Joystick Control

- **Details**: Inspired by the move-it joystick node, tested with an XBOX joystick.
- **Launch Instructions**:
```
roslaunch panda_gazebo panda_rviz.launch
rosrun panda_gazebo joystick_teleop.py
```
- **Data Saving**: Option to save the robot's states/velocities as numpy arrays for later cVAE training.

## cVAE Teleoperation

### Running the Sinus Demo

- **Steps
```
roslaunch panda_gazebo panda_rviz.launch
rosrun panda_gazebo plot_sine_wave.py
rosrun panda_gazebo cvae_keyboard_teleop.py
```
### Training cVAE on a New Task

1. Follow the manual teleoperation steps to generate training data.
2. Modify `pandas_data_parser.py` to process the new numpy files with your custom experiment data.
3. Adjust the latent dimension in `train_final.py` according to the specific task you are addressing. Also, update the cVAE model name to reflect the task or dataset you are using.
4. Run the following command to start the training process. The best model, based on validation loss (split as 80% training data and 20% validation data), will be saved in `catkin_ws/cvae_model`:
```
python train_final.py
```
5. Update `catkin_ws/src/panda_gazebo/scripts/cvae_keyboard_teleop.py` to ensure that the latent dimension matches the trained model. Also, verify that the cVAE model name is correctly specified.

- Use the keyboard shortcuts `(a q)` to control the first dimension of the latent action and `(z s)` for the second dimension. 
- For additional latent dimensions, you can define your custom keyboard shortcuts by following the examples provided in lines 196 to 199 of the script.

6. To initiate the cVAE-based teleoperation with the newly trained model, execute:
```
roslaunch panda_gazebo panda_rviz.launch
rosrun panda_gazebo cvae_keyboard_teleop.py
```
This procedure enables the teleoperation of the Franka Emika Robot using a cVAE model trained on new tasks, facilitating the exploration of various applications and enhancements in robotic teleoperation through interactive learning models.

