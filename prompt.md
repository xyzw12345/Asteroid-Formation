# Help on coding about n-body simulation

Please help me with the following task:

I would like to write a project about the formation of asteroids, to be more specifically, my theoretical model is that the solar system has only the sun in the center and a large number of small objects in the orbits (like the asteroid belt in reality). Under customized initial condition (which I hope is adjustable), I hope to simulate the motion of this dynamic system and merge two objects when a collision occurred.

Throughout the process, we can assume that the objects are spherical, and they merge when collisions occur.

## Requirements

* I would like the project to be in C++ or python (so that I'm more familiar with the language), but as fast as possible

* I would like the code to be structured and easy to maintain, the implementations of the specific parts described below can be easily changable

* I hope that the project can be fast, e.g. simulating multiple steps of over 50,000 objects on my laptop

* The project can make use of Nvidia GPU (e.g. CUDA programming) if this can make the project faster.

## Specific description

From my perspective, the task can be decomposed into several parts:

* how to compute the accelerations at each moment efficiently

* how to detect collisions at each moment

* An implementation of visualization, e.g. plot the objects as glittering particles and color them according to weight and speed

You can first analyze the whole task, then give me a more detailed step-by-step job description, so that we can collaborate better later on. Thanks!



# Confirmation on details of the project

* My computer: Windows 11, CPU i9-14900HX, GPU RTX-4060, but can switch to Linux Mint 22 if necessary

* How do you anticipate the python hybrid approach will perform? Will data transmittion between different parts of the code be rather slow? e.g. from python to integrated C++ part.


# Phase 0: Setup and Basic Data Structures

Can you help me implement the plans of this stage? I already have a new blank github repo on this, so we can start from the code.

The Structure of Arrays (SoA) approach is prefered, as long as

* it doesn't affect modularity

* it works well under the condition that the number of particles may change over time (since we want to handle collisions)

I prefer that the code is available for running and testing at every intermediate step, i.e., we can start with a really trivial implementation (maybe O(n^2) and on CPU) for testing, to make sure that everything works as expected.

Thanks!



# Thoughts on Phase 0

I run the code locally, and found out about the following phenomenon: 

When two particles are supposed to collide, there will be one step that they come really close, and therefore gain a rather large acceleration. So after that step, they'll get a large velocity and just fly pass each other, which not only doesn't fulfill our goal, but also fails to show the preservation of energy.

How would you try to handle this phenomenon? Could we try some sort of adjustable step size? Or determine earlier when they'll collide? Please also suggest some thoughts of yours on this subject, and demonstrate how we could fix this bug.

Thanks!



# Another thought on visualization

I'm wondering if we could make the visualization to be an animation, where a new frame represents one (or several) step forward, and the user can pause/resume and drag the perspective for the 3-dimensional animation.

Would it be possible for you to implement such an idea? You can use any package available as you want, but please keep in mind the performance requirement that we're going to eventually scale up to 50000 particles.