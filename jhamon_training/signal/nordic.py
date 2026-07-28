"""Vendored from the jhamon base library (signal/nordic.py, commit d1d7d3b).

Copied verbatim on 2026-05-30 to make jhamon-training self-contained and give
the manuscript a single source of truth (no external editable git dependency).
Contains: dame_inercia (knee inertia for NH inertia correction) and its helper
_segmentCMaxes. If the upstream version changes, update this copy deliberately.
"""


    
def dame_inercia(REP, bm, ex_weight):
    
    """Get a vector of len(REP) samples with the moment of inertia (kg*m2). This function takes into account some reference values 
    (taken from Enoka's book) to first find the location of the system CM and then calculate the moment of inertia. 
    
    Parameters:
    --------------
    REP: array of data containing force and kinematic data of a given nordic hamstring repetition. 
    bw: body mass (kg)
    
    Returns:
    ----------
    I_knee: moment of inertia around the knee angle. 
    
    Notes
    ----------
    The total moment of inertia is applied to each force data (from left and right leg).
    
    """
    import numpy as np
    
    # Some previous values needed
    body_mass = bm # import from file for each subject
    gravity = 9.8
    body_weight = body_mass * gravity
    
    # Estimations of segments and system weights
    thigh_weight = 0.127 * body_weight - 14.82
    trunk_weight = 0.532 * body_weight - 6.93
    head_weight = 0.032 * body_weight + 18.70
    upperarm_weight = 0.022 * body_weight + 4.76
    forearm_weight = 0.013 * body_weight + 2.41
    hand_weight = 0.005 * body_weight + 0.75
    arm_weight = upperarm_weight + forearm_weight + hand_weight
    extra_weight = ex_weight * 9.8 # import from file
    trunk_complete_weight = trunk_weight + head_weight + (arm_weight)*2 + extra_weight
    system_weight = thigh_weight + trunk_complete_weight + extra_weight
    
    # Masses
    thigh_mass = thigh_weight / gravity
    trunk_mass = trunk_weight / gravity
    head_mass = head_weight / gravity
    upperarm_mass = upperarm_weight / gravity
    forearm_mass = forearm_weight / gravity
    hand_mass = hand_weight / gravity
    extra_mass = extra_weight / gravity
    trunk_complete_mass = trunk_complete_weight / gravity
    system_mass = system_weight / gravity
    # CM locations for trunk and thigh segments 
    head_length = 20.33 # from reference values
    head_CM = 0.597 * head_length 

    I_knee = np.zeros((len(REP)))
    for ii in np.arange(len(REP)):
        # Define trunk and sum the segmental torques about the y and x axis : 
        trunk_axes = np.array(((REP[ii,9], REP[ii,10]), (REP[ii,13], REP[ii,14])))
        trunk_CM_axes = _segmentCMaxes(trunk_axes, 0.522)
        trunk_weight_torque_x = trunk_complete_weight * (trunk_CM_axes[0])
        trunk_weight_torque_y = trunk_complete_weight * (trunk_CM_axes[1])
        
        # Same for the thigh segment
        thigh_axes = np.array(((REP[ii,13], REP[ii,14]), (REP[ii,15], REP[ii,16])))
        thigh_CM_axes = _segmentCMaxes(thigh_axes, 0.398)
        thigh_weight_torque_x = trunk_complete_weight * (thigh_CM_axes[0])
        thigh_weight_torque_y = trunk_complete_weight * (thigh_CM_axes[1])
        
        # Resultant weight torques
        f_x = sum((trunk_weight_torque_x, thigh_weight_torque_x))
        f_y = sum((trunk_weight_torque_y, thigh_weight_torque_y))
        
        # Get CM positions (in the global coordinate system provided)
        CM_axes = [f_x / system_weight, f_y / system_weight]
    
        # Distance from CM to knee joint 
        CM_distance_knee = np.sqrt( (CM_axes[0] - REP[ii,15])**2 + (CM_axes[1] - REP[ii,16])**2 )
        thighCM_dist = np.sqrt( (thigh_CM_axes[0] - CM_axes[0])**2 + (thigh_CM_axes[1] - CM_axes[1])**2 )
        trunkCM_dist = np.sqrt( (trunk_CM_axes[0] - CM_axes[0])**2 + (trunk_CM_axes[1] - CM_axes[1])**2 )
        
        # Inertia around the CM
        I_g = (thigh_mass*2)*(thighCM_dist**2) + trunk_complete_mass*(trunkCM_dist**2) # Kg.m2
    
        # Inertia around the center of rotation (knee)
        I_knee[ii] = I_g + (system_mass*(CM_distance_knee**2)) # k.m2
    
    return(I_knee)


def _segmentCMaxes(segment_axes, cm_pos):
    
    """Get the coordinates (x,y) of the center of mass of a given segment.
    
    Parameters:
    --------------
    segment_axes: 2x2 array containing x and y axes of two points defining a given segment. Column 1 and 2 are x and y values, respectively.
    cm_pos: percentage number indicating the position of the center of mass, relative to segment length
    
    Returns:
    ----------
    List with coordinates (x,y) of the center of mass (CM)
    
    Notes
    ----------
    'cm_pos' values can be estimated from reference values provided by models (See Enoka book) 
    
    """
    x_CM = ((segment_axes[0,0] - segment_axes[1,0]) * cm_pos) + segment_axes[1,0]
    y_CM = ((segment_axes[0,1] - segment_axes[1,1]) * cm_pos) + segment_axes[1,1]
    
    # plt.plot(segment_axes[:,0], segment_axes[:,1], 'ro')
    # plt.plot(x_CM, y_CM, 'go')
    
    return([x_CM, y_CM])