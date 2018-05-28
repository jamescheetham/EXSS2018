"""
DataSet contains a dict of samples keyed by the sample id
A Sample contains a dict of joints keyed by the joint name
A Joint contains a dict of JointData keyed by the frame number
A JointData contains multidimensional dict keyed by the data type (position or velocity) and a plane (x or y) as well
as angle data (if relevant).
"""
from xml.etree import ElementTree as Et
import sys
import csv
import os
import matplotlib.pyplot as plt


class DataSet:
    def __init__(self, xml_config_file):
        self.xml_config_file = xml_config_file
        self.xml = None
        self.xml_root = None
        self.samples = {}
        self.phases = {}
        self.graphs = []
        self.extract_data = []
        self.process_config_file()

    def process_config_file(self):
        self.xml = Et.parse(self.xml_config_file)
        self.xml_root = self.xml.getroot()
        definitions = self.xml_root.find('definitions')
        self.generate_phases(definitions.find('phases'))
        self.generate_graph_data(self.xml_root.find('analysis'))
        self.generate_extract_data(self.xml_root.find('analysis'))
        for d in self.xml_root.findall('dataset'):
            ds_type = d.get('type')
            try:
                ds_id = int(d.get('id'))
            except ValueError:
                ds_id = -1
            if ds_id == -1 or ds_type is None:
                sys.exit('There is an invalid XML Entry for a Dataset')
            new_sample = Sample(ds_type, d, ds_id)
            self.samples.update({ds_id: new_sample})
            """
            kp = d.find('keypoints')
            for p in kp.findall('point'):
                kp_name = p.get('name')
                if kp_name is None:
                    sys.exit('The Dataset with ID %d has a Key Point with no name' % ds_id)
                if kp_name not in new_sample.key_points:
                    sys.exit('The Dataset with ID %d has a Key Point that is not defined (%s)'
                             % (ds_id, kp_name))
                try:
                    framenum = int(p.get('framenum'))
                except ValueError:
                    sys.exit('The Sample %d has a Key Point %s with an invalid frame number (%s)'
                             % (ds_id, kp_name, p.get('framenum')))
                except TypeError:
                    sys.exit('The Sample %d has a Key Point %s with no frame number'
                             % (ds_id, kp_name))
                self.samples[ds_id].set_keypoint(kp_name, framenum)
            """

    def generate_phases(self, phases):
        for p in phases.findall('phase'):
            name = p.get('name')
            start = p.get('start')
            end = p.get('end')
            if name is None or start is None or end is None:
                sys.exit('A Phase is defined without a Name, Start Point or End Point')
            self.phases.update({name.replace(' ', '_'): Phase(start, end)})

    def generate_graph_data(self, analysis_data):
        for g in analysis_data.findall('graph'):
            self.graphs.append(Graph(g, self.phases))

    def generate_extract_data(self, analysis_data):
        for d in analysis_data.findall('data'):
            self.extract_data.append(ExtractData(d, self.phases))

    def get_key_point_data(self, sample_id, keypoint):
        if sample_id not in self.samples:
            sys.exit('Requested Key Point Data for sample id %d which does not exist' % sample_id)
        return self.samples[sample_id].get_keypoint_data(keypoint)

    def create_graphs(self):
        for g in self.graphs:
            fig = plt.figure(figsize=(10, 10))
            plot = fig.add_subplot(1, 1, 1)
            plot.set_title(g.title)
            plot.set_xlabel(g.x_axis_title)
            plot.set_ylabel(g.y_axis_title)
            for k, v in self.samples.items():
                v.graph(g, plot)
            fig.savefig(g.filename)

    def process_extract_data(self):
        for d in self.extract_data:
            pass


class Sample:
    def __init__(self, sample_type, data_set_info, sample_id):
        self.sample_id = sample_id
        self.sample_type = sample_type
        self.joints = {}
        self.key_points = {}
        for file_info in data_set_info.findall('file'):
            self.process_file(file_info)
        for j in self.joints.items():
            for d in j:
                print(d)
        kp = data_set_info.find('keypoints')
        for p in kp.findall('point'):
            self.process_key_points(p)

    def process_file(self, file_info):
        filename = file_info.get('filename')
        if filename is None:
            sys.exit('There is no filename specified for a file entry')
        if not os.path.isfile(filename):
            sys.exit('The file %s does not exist' % filename)
        try:
            framerate = int(file_info.get('framerate'))
        except ValueError:
            sys.exit('The file %s has an invalid framerate' % filename)
        y_fudge = file_info.get('y-fudge')
        if y_fudge is None:
            y_fudge = 0
        else:
            try:
                y_fudge = float(file_info.get('y-fudge'))
            except ValueError:
                sys.exit('The file %s has an invalid y-fudge value' % filename)
        with open(filename, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t', quotechar='"')
            if file_info.get('skip') is not None:
                try:
                    skip_count = int(file_info.get('skip'))
                except ValueError:
                    sys.exit('Invalid Skip value for file %s' % filename)
                for i in range(skip_count):
                    next(csv_reader)
            columns = Sample.get_column_data(file_info, filename)
            line_count = 0
            for line in csv_reader:
                for c in columns:
                    if c.joint not in self.joints:
                        self.joints.update({c.joint: Joint(c.joint)})
                    try:
                        self.joints[c.joint].update_joint(c, line[c.col_num - 1], line_count, framerate, y_fudge)
                    except ValueError:
                        sys.exit("There was an error processing the data in column %d for file %s for frame number %d."
                                 % (c.col_num, filename, line_count + 1))
                line_count += 1

    def process_key_points(self, point_data):
        self.key_points.update({point_data.get('name'): KeyPoint(point_data)})

    @staticmethod
    def get_column_data(file_info, filename):
        columns = file_info.find('columns')
        cols = columns.findall('col')
        column_data = [None] * len(cols)
        for c in cols:
            tmp_col = ColumnData(c, filename)
            if column_data[tmp_col.col_num - 1] is not None:
                sys.exit('The Column Number %d for file %s is duplicated' % (tmp_col.col_num, filename))
            column_data[tmp_col.col_num - 1] = tmp_col
        if None in column_data:
            sys.exit('There was an error processing Column Data for the file %s' % filename)
        return column_data

    def graph(self, graph_data, fig):
        if graph_data.joint in self.joints:
            phase_start = graph_data.phase.start
            phase_end = graph_data.phase.end
            if phase_start not in self.key_points or phase_end not in self.key_points:
                sys.exit("Key Point %s or %s not defined for Sample ID %d"
                         % (phase_start, phase_end, self.sample_id))
            x_data = []
            y_data = []
            self.joints[graph_data.joint].generate_graph_data(x_data, y_data, self.key_points[phase_start],
                                                              self.key_points[phase_end], graph_data)
            if graph_data.graph_type == 'xy_plot':
                fig.plot(x_data, y_data, marker='o', markersize=8)
            else:
                fig.plot(x_data, y_data)


class ColumnData:
    VALID_TYPES = ['position', 'angle', 'velocity']
    TYPE_HAS_COORDS = ['position', 'velocity']
    VALID_PLANES = ['x', 'y']

    def __init__(self, col_data, filename):
        try:
            self.col_num = int(col_data.get('colnum'))
        except ValueError:
            sys.exit('Invalid Column Number value for file %s' % filename)
        except TypeError:
            sys.exit('A Column for file %s does not have a column number' % filename)
        self.type = col_data.get('type')
        if self.type is None:
            sys.exit('The Column Number %d for file %s does not have a type specified' % (self.col_num, filename))
        self.type = self.type.lower()
        if self.type not in ColumnData.VALID_TYPES:
            sys.exit('The Column Number %d for file %s does not have a valid type (should be %s)' %
                     (self.col_num, filename, ', '.join(ColumnData.VALID_TYPES)))
        self.joint = col_data.get('joint')
        if self.joint is None:
            sys.exit('The Column Number %d for file %s does not have a joint specified' % (self.col_num, filename))
        if self.type in ColumnData.TYPE_HAS_COORDS:
            self.plane = col_data.get('plane')
            if self.plane is None:
                sys.exit('The Column Number %d for file %s does not have a plane specified' % (self.col_num, filename))
            self.plane = self.plane.lower()
            if self.plane not in ColumnData.VALID_PLANES:
                sys.exit('The Column Number %d for file %s does not have a valid plane specified (should be %s)' %
                         (self.col_num, filename, ', '.join(ColumnData.VALID_PLANES)))


class Joint:
    def __init__(self, joint_name):
        self.joint_name = joint_name
        self.joint_data = {}

    def update_joint(self, column_data, data, frame_num, framerate, y_fudge):
        if frame_num not in self.joint_data:
            self.joint_data.update({frame_num: JointData(frame_num, framerate)})
        self.joint_data[frame_num].add_joint_data(column_data, data, y_fudge)

    def __str__(self):
        output_str = 'Joint Name: %s' % self.joint_name
        for j in self.joint_data.values():
            output_str += str(j)
        return output_str

    def generate_graph_data(self, x_data, y_data, phase_start, phase_end, graph_data):
        x_data_type = graph_data.x_axis.split(' ')
        y_data_type = graph_data.y_axis.split(' ')
        if graph_data.normalise:
            Joint.normalise_data(x_data, phase_start, phase_end)
        else:
            for i in range(phase_start.frame_num, phase_end.frame_num+1):
                x_data.append(self.joint_data[i].get(x_data_type[1], x_data_type[0]))
        for i in range(phase_start.frame_num, phase_end.frame_num+1):
            y_data.append(self.joint_data[i].get(y_data_type[1], y_data_type[0]))

    @staticmethod
    def normalise_data(x_data, phase_start, phase_end):
        phase_length = phase_end.frame_num - phase_start.frame_num
        for i in range(phase_start.frame_num, phase_end.frame_num+1):
            x_data.append((i - phase_start.frame_num)/phase_length*100)


class JointData:
    def __init__(self, frame_num, framerate):
        self.frame_num = frame_num
        self.timestamp = frame_num * 1 / framerate
        self.data = {i: {j: None for j in ColumnData.VALID_PLANES} for i in ColumnData.TYPE_HAS_COORDS}
        self.angle = None

    def add_joint_data(self, column_data, data, y_fudge):
        if column_data.type in ColumnData.TYPE_HAS_COORDS:
            self.data[column_data.type][column_data.plane] = float(data) + (self.frame_num - 1) * y_fudge
        else:
            self.angle = float(data)

    def get(self, data_type, plane):
        if data_type in ColumnData.VALID_TYPES:
            if data_type in ColumnData.TYPE_HAS_COORDS:
                if plane in ColumnData.VALID_PLANES:
                    return self.data[data_type][plane]
            else:
                return self.angle
        return None

    def __str__(self):
        output_str = '\nTimestamp: %0.2f' % self.timestamp
        for k1, v1 in self.data.items():
            for k2, v2 in v1.items():
                if v2 is not None:
                    output_str += '\n%s %s: %0.3f' % (k2.upper(), k1.capitalize(), v2)
        if self.angle is not None:
            output_str += '\nAngle: %0.2f' % (0 if self.angle is None else self.angle)
        return output_str


class KeyPoint:
    def __init__(self, point_data):
        self.name = point_data.get('name')
        self.frame_num = int(point_data.get('framenum'))


class Phase:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class Graph:
    def __init__(self, xml_data, phases):
        self.filename = xml_data.get("file")
        self.title = xml_data.get("title")
        self.joint = xml_data.get("joint")
        self.x_axis = xml_data.get("x-axis")
        self.y_axis = xml_data.get("y-axis")
        if xml_data.get("phase") is None:
            sys.exit("No Phase specified in Graph Definition for %s" % self.title)
        phase_key = xml_data.get("phase").replace(" ", "_")
        if phase_key in phases:
            self.phase = phases[phase_key]
        else:
            sys.exit("Unknown Phase specified in Graph titled %s" % self.title)
        self.x_axis_title = xml_data.get("x-axis-title")
        self.y_axis_title = xml_data.get("y-axis-title")
        self.normalise = (xml_data.get('normalise') == 'true')


class ExtractData:
    def __init__(self, xml_data, phases):
        self.name = xml_data.get("name")
        self.plane = xml_data.get("plane")
        self.file_name = xml_data.get("filename")
        self.value = xml_data.get("value")
        self.joint1 = xml_data.get("joint1")
        self.joint2 = xml_data.get("joint2")
        self.point = xml_data.get("point")
        self.joint = xml_data.get("joint")
        self.phase = xml_data.get("phase")
        if self.joint1 is None and self.joint is None:
            sys.exit("A Defined Data (%s) does not have sufficient information", self.name)
        if self.point is None and self.phase is None:
            sys.exit("A Defined Data (%s) does not have sufficient information", self.name)
        if self.phase is not None and self.phase not in phases:
            sys.exit("A Defined Data (%s) does not have a valid phase", self.name)


def main():
    ds = DataSet('assignment.xml')
    ds.process_extract_data()
    ds.create_graphs()


if __name__ == '__main__':
    main()
