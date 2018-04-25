import csv
import matplotlib.pyplot as plt


class JointData:
    def __init__(self, joint_name, position_offset):
        self.joint_name = joint_name
        self.position_offset = position_offset
        self.timeslices = []
        self.has_angle_data = False

    def add_timeslice(self, frame_num, line_data, fudge_factor):
        self.timeslices.append(TimeSlice(frame_num, line_data, self.position_offset, fudge_factor))

    def add_position_data_to_plot(self, fig):
        x_pos = []
        y_pos = []

        for t in self.timeslices:
            t.add_to_pos_array(x_pos, y_pos)
        fig.plot(x_pos, y_pos)

    def add_angle_data(self, line, offset, framenum):
        self.has_angle_data = True
        if framenum < len(self.timeslices):
            self.timeslices[framenum-1].z_ang = float(line[offset])

    def add_velocity_data_to_plot(self, fig, axis_dir):
        x_data = []
        y_data = []

        for t in self.timeslices:
            t.add_to_vel_array(x_data, y_data, axis_dir)
        fig.plot(x_data, y_data)

    def add_angle_to_plot(self, fig):
        z_ang = []

        for t in self.timeslices:
            z_ang.append(t.z_ang)
        fig.plot(z_ang)
        
    def add_xy_data(self, x_data, y_data, framenum):
      if framenum - 1 < len(self.timeslices):
        self.timeslices[framenum-1].add_to_pos_array(x_data, y_data)

    def __str__(self):
        output_str = "Joint: %s" % self.joint_name
        for t in self.timeslices:
            output_str += '\n%s' % t
        return output_str


class TimeSlice:
    def __init__(self, frame_num, line_data, offset, fudge_factor):
        self.frame_num = frame_num
        self.x_pos = float(line_data[offset])
        self.y_pos = float(line_data[offset+1]) + (frame_num - 1) * fudge_factor
        self.x_vel = float(line_data[offset+2])
        self.y_vel = float(line_data[offset+3])
        self.z_ang = None

    def add_to_pos_array(self, x_pos, y_pos):
        x_pos.append(self.x_pos)
        y_pos.append(self.y_pos)

    def add_to_vel_array(self, x_data, y_data, axis_dir):
        x_data.append(self.frame_num)
        y_data.append(self.x_vel if axis_dir.lower() == 'x' else self.y_vel)

    def add_to_ang_array(self, z_ang):
        z_ang.append(self.z_ang)

    def __str__(self):
        return "Frame: %d, X Pos: %f, Y Pos: %f, X Vel: %f, Y Vel: %f" % (self.frame_num, self.x_pos, \
                                                                          self.y_pos, self.x_vel, self.y_vel)


def main():
    framecount = None
    #fudge_factor = -0.0019024 #Shod
    fudge_factor = -0.00190476190476 #UnShod
    joints = [JointData('glenohumeral', 0), JointData('hip', 4), JointData('knee', 8), JointData('ankle', 12), JointData('toe', 16)]
    with open('UnShod-Linear3.txt', 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t', quotechar='"')
        for i in range(6):
            next(csv_reader)
        i = 1
        for line in csv_reader:
            for j in joints:
                j.add_timeslice(i, line, fudge_factor)
            i += 1
        framecount = i

    ankle = None
    knee = None
    for j in joints:
        if j.joint_name == 'ankle':
            ankle = j
        if j.joint_name == 'knee':
            knee = j
        if j.joint_name == 'hip':
            hip = j
    with open('UnShod-Ang3.txt', 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t', quotechar='"')
        for i in range(6):
            next(csv_reader)
        i = 1
        for line in csv_reader:
            ankle.add_angle_data(line, 2, i)
            knee.add_angle_data(line, 1, i)
            hip.add_angle_data(line, 0, i)
            i += 1

    pos_fig = plt.figure(figsize=(10, 10))
    pos_fig_plot = pos_fig.add_subplot(1, 1, 1)

    for j in joints:
        fig = plt.figure(figsize=(10, 10))
        pos_plot = fig.add_subplot(4, 1, 1)
        vel_x_plot = fig.add_subplot(4, 1, 2)
        vel_y_plot = fig.add_subplot(4, 1, 3)
        if j.has_angle_data:
            ang_z_plot = fig.add_subplot(4, 1, 4)
            ang_z_plot.set_title('Z Angle')
        pos_plot.set_title('Position')
        vel_x_plot.set_title('X Velocity')
        vel_y_plot.set_title('Y Velocity')
        j.add_position_data_to_plot(pos_plot)
        j.add_velocity_data_to_plot(vel_x_plot, 'x')
        j.add_velocity_data_to_plot(vel_y_plot, 'y')
        j.add_position_data_to_plot(pos_fig_plot)
        if j.has_angle_data:
            j.add_angle_to_plot(ang_z_plot)
        fig.savefig('%s - Graph.png' % j.joint_name.capitalize())

    pos_fig.savefig('Position Graph.png')
    
    xy_plot = plt.figure(figsize=(30, 10))
    xy_plot_fig = xy_plot.add_subplot(1, 1, 1)
    
    for i in range(1, framecount, 1):
      x_data = []
      y_data = []
      for j in joints:
        j.add_xy_data(x_data, y_data, i)
      xy_plot_fig.plot(x_data, y_data, marker='o', markersize=8)
    
    xy_plot.savefig('XY Plot.png')


if __name__ == '__main__':
    main()
