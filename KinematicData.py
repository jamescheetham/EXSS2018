from xml.etree import ElementTree as Et
import numpy as np
import sys
import csv
import os
import matplotlib.pyplot as plt
import math
from optparse import OptionParser

"""
DataSet contains a dict of samples keyed by the sample id
A Sample contains a dict of joints keyed by the joint name
A Joint contains a dict of JointData keyed by the frame number
A JointData contains multidimensional dict keyed by the data type (position or velocity) and a plane (x or y) as well
as angle data (if relevant).
"""

"""
Need to move Column Data to be under dataset for the XML file, then all files will use that.

The program needs to import the calibration data and determine the neutral values. The angle
data may require calculation, as it needs to be determined from the values of the joints on
either side.

Need to also tell the program what is the "floor" value. This is the lowest level that the marker will appear.
This will allow the shod and unshod to have the same 0 marker.

For the calibration data, it will need to be averaged across all of the information that is present in the file
However, if there are two calibration files, then the information will need to be averaged between them as well.
THis means that the calibration information needs to be in a list.

I need to get legend information onto the graphs, and write the system that is able to extract data
to get absolute values (i.e. distance between two markers at a particular phase).

Processing Order
For each Data Set
 - Read Column Information - Done
 - For each Calibration File - Done
  - Read and store
 - Average Calibration Information
 - Read Data File normalising against calibration information and fudge factor

"""


class KinematicData:
    def __init__(self, xml_config_file):
        self.xml_config_file = xml_config_file
        self.xml = None
        self.xml_root = None
        self.samples = {}
        self.phases = {}
        self.graphs = []
        self.extract_data = []
        self.extract_data_filename = None
        self.segments = []
        self.process_config_file()

    def process_config_file(self):
        self.xml = Et.parse(self.xml_config_file)
        self.xml_root = self.xml.getroot()
        definitions = self.xml_root.find('definitions')
        analysis = self.xml_root.find('analysis')
        self.generate_phases(definitions.find('phases'))
        self.generate_graph_data(analysis.find('graphs'))
        self.generate_extract_data(analysis.find('data'))
        self.generate_segment_data(definitions.find('segments'))
        for d in self.xml_root.findall('dataset'):
            ds_type = d.get('type')
            try:
                ds_id = int(d.get('id'))
            except ValueError:
                ds_id = -1
            if ds_id == -1 or ds_type is None:
                sys.exit('There is an invalid XML Entry for a Dataset')
            new_sample = Sample(ds_type, d, ds_id, d.get('name'))
            self.samples.update({ds_id: new_sample})

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
        self.extract_data_filename = analysis_data.get('file')
        if self.extract_data_filename is None:
            sys.exit("The Analysis Data does not have a File Name specified")
        for d in analysis_data.findall('row'):
            self.extract_data.append(ExtractData(d, self.phases))

    def generate_segment_data(self, xml_data):
        for s in xml_data.findall('segment'):
            self.segments.append(Segment(s))

    def get_key_point_data(self, sample_id, keypoint):
        if sample_id not in self.samples:
            sys.exit('Requested Key Point Data for sample id %d which does not exist' % sample_id)
        return self.samples[sample_id].get_keypoint_data(keypoint)

    def create_graphs(self):
        for g in self.graphs:
            legend_titles = []
            fig = plt.figure(figsize=(10, 10))
            plot = fig.add_subplot(1, 1, 1)
            plot.set_title(g.title)
            plot.set_xlabel(g.x_axis_title)
            plot.set_ylabel(g.y_axis_title)
            x_axis_min = None
            x_axis_max = None
            for k, v in self.samples.items():
                legend_titles.append(v.name)
                tmp_min, tmp_max = v.graph(g, plot)
                if x_axis_min is None or tmp_min < x_axis_min:
                    x_axis_min = tmp_min
                if x_axis_max is None or tmp_max > x_axis_max:
                    x_axis_max = tmp_max
            plot.legend(legend_titles)
            # noinspection PyTypeChecker
            plot.set_xlim([x_axis_min, x_axis_max])
            fig.savefig(g.filename)

    def process_extract_data(self):
        with open(self.extract_data_filename, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = ['title']
            for s in self.samples.values():
                header.append('%d - %s - %s' % (s.sample_id, s.sample_type, s.name))
            csv_writer.writerow(header)
            for e in self.extract_data:
                e.do_extract(csv_writer, self.samples, self.phases)

    def get_segment_lengths(self):
        for s in self.samples.values():
            s.write_segment_lengths(self.segments)

    def raw_export_data(self):
        for s in self.samples.values():
            s.raw_export_data()


class Sample:
    def __init__(self, sample_type, data_set_info, sample_id, name):
        self.sample_id = sample_id
        self.sample_type = sample_type
        self.name = name
        self.joints = {}
        self.key_points = {}
        self.columns = {}
        self.calibration_data = []  # will be a list of dicts that contain the calibration data
        self.calibration_zeros = {}  # will be a list of dicts that contain baseline data
        self.frame_count = 0
        self.process(data_set_info)

    def process(self, data_set_info):
        self.generate_column_data(data_set_info.find('columns'), self.sample_id)
        print("Generated Column Data")
        self.get_calibration_data(data_set_info.findall('calibration'))
        print("Pulled Calibration Data")
        low_point = self.calc_calibration_zeros(data_set_info.find('normalisation'))
        datafiles = data_set_info.find('datafile')
        for details in datafiles.findall('details'):
            file_info = FileInfo(details, self.sample_id)
            self.process_datafile(file_info, low_point)
            print("Processed DataFile %s" % file_info.filename)
        kp = data_set_info.find('keypoints')
        for p in kp.findall('point'):
            self.process_key_points(p)

    def get_calibration_data(self, xml_calibration):
        for cali in xml_calibration:
            tmp_joint_data = {}
            for c in cali.findall('details'):
                file_info = FileInfo(c, self.sample_id)
                with open(file_info.filename, 'r') as f:
                    csv_reader = csv.reader(f, delimiter='\t', quotechar='"')
                    for _ in range(file_info.skip_count):
                        next(csv_reader)
                    self.read_file(csv_reader, file_info, tmp_joint_data)
            self.calibration_data.append(tmp_joint_data)

    def calc_calibration_zeros(self, xml_data):
        averages = []
        for c in self.calibration_data:
            tmp_data = {}
            for k, v in c.items():
                tmp_data.update({k: v.calc_averages()})
            averages.append(tmp_data)
        if len(averages) > 1:
            tmp_joint_data = {}
            for a in averages:
                for k, v in a.items():
                    if k not in tmp_joint_data:
                        tmp_joint_data.update({k: JointData(1, 1)})
                    tmp_joint_data[k] += v
            for k, v in tmp_joint_data.items():
                self.calibration_zeros.update({k: v / len(averages)})
        else:
            self.calibration_zeros = averages[0]
        if xml_data is not None:
            low_joint_xml = xml_data.find('low_point')
            if low_joint_xml is not None:
                low_joint_name = low_joint_xml.get('joint')
                if low_joint_name is None or low_joint_name not in self.calibration_zeros:
                    sys.exit('The Calibration Data specifies a Joint for low_point for Sample %d \
                    that is not defined (%s)'
                             % (self.sample_id, low_joint_name))
                return self.calibration_zeros[low_joint_name].get('position', 'y')
        return 0

    def read_file(self, csv_reader, file_info, joint_data, low_point=0):
        line_count = 1
        for line in csv_reader:
            for c in self.columns[file_info.data_type]:
                if c.joint not in joint_data:
                    joint_data.update({c.joint: Joint(c.joint)})
                try:
                    joint_data[c.joint].update_joint(c, line[c.col_num - 1], line_count,
                                                     file_info.framerate, file_info.y_fudge, low_point)
                except ValueError:
                    sys.exit("There was an error processing the file %s in column %d for frame number %d" %
                             (file_info.filename, c.col_num, line_count + 1))
            line_count += 1
        return line_count - 1

    def process_datafile(self, file_info, low_point):
        with open(file_info.filename, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t', quotechar='"')
            for _ in range(file_info.skip_count):
                next(csv_reader)
            self.frame_count = self.read_file(csv_reader, file_info, self.joints, low_point)

    def process_key_points(self, point_data):
        self.key_points.update({point_data.get('name'): KeyPoint(point_data)})

    def generate_column_data(self, xml_columns, dataset_id):
        for column_type in xml_columns:
            self.columns.update({column_type.tag: []})
            cols = column_type.findall('col')
            self.columns[column_type.tag] = [None] * len(cols)
            for c_data in cols:
                tmp_col = ColumnData(c_data, dataset_id)
                if self.columns[column_type.tag][tmp_col.col_num - 1] is not None:
                    sys.exit('The Column Number %d for file %s is duplicated' % (tmp_col.col_num, dataset_id))
                self.columns[column_type.tag][tmp_col.col_num - 1] = tmp_col
            if None in self.columns[column_type.tag]:
                sys.exit('There was an error processing Column Data for the file %s' % dataset_id)

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
            fig.plot(x_data, y_data)
            return min(x_data), max(x_data)
        return 0, 0

    def write_segment_lengths(self, segments):
        filename = 'segment_lengths_%s.csv' % self.name
        with open(filename, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['frame number'] + [x.name for x in segments])
            for i in range(1, self.frame_count + 1):
                tmp_array = [i]
                for s in segments:
                    tmp_array.append(s.get_length(self, i))
                csv_writer.writerow(tmp_array)

    def raw_export_data(self):
        filename = 'data_export_%s.csv' % self.name
        with open(filename, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            base_header = ColumnData.generate_header_row()
            header = []
            for j in self.joints.keys():
                for h in base_header:
                    header.append('%s %s' % (j, h))
            csv_writer.writerow(['framenumber'] + header)
            output_list = ['calibration']
            for v in header:
                tmp = v.split(' ')
                if len(tmp) == 2:
                    output_list.append(self.calibration_zeros[tmp[0]].angle)
                else:
                    output_list.append(self.calibration_zeros[tmp[0]].data[tmp[2]][tmp[1]])
            csv_writer.writerow(output_list)
            for i in range(1, self.frame_count + 1):
                output_list = [i + 1]
                for v in header:
                    tmp = v.split(' ')
                    if len(tmp) == 2:
                        output_list.append(self.joints[tmp[0]].joint_data[i].angle)
                    else:
                        output_list.append(self.joints[tmp[0]].joint_data[i].data[tmp[2]][tmp[1]])
                csv_writer.writerow(output_list)


class ColumnData:
    VALID_TYPES = ['position', 'angle', 'velocity']
    TYPE_HAS_COORDS = ['position', 'velocity']
    VALID_PLANES = ['x', 'y']

    def __init__(self, col_data, dataset_id):
        try:
            self.col_num = int(col_data.get('colnum'))
        except ValueError:
            sys.exit('Invalid Column Number value for dataset ID %d' % dataset_id)
        except TypeError:
            sys.exit('A Column for file %s does not have a column number' % dataset_id)
        self.type = col_data.get('type')
        if self.type is None:
            sys.exit('The Column Number %d for file %s does not have a type specified' % (self.col_num, dataset_id))
        self.type = self.type.lower()
        if self.type not in ColumnData.VALID_TYPES:
            sys.exit('The Column Number %d for file %s does not have a valid type (should be %s)' %
                     (self.col_num, dataset_id, ', '.join(ColumnData.VALID_TYPES)))
        self.joint = col_data.get('joint')
        if self.joint is None:
            sys.exit('The Column Number %d for file %s does not have a joint specified' % (self.col_num, dataset_id))
        if self.type in ColumnData.TYPE_HAS_COORDS:
            self.plane = col_data.get('plane')
            if self.plane is None:
                sys.exit('The Column Number %d for file %s does not have a plane specified' %
                         (self.col_num, dataset_id))
            self.plane = self.plane.lower()
            if self.plane not in ColumnData.VALID_PLANES:
                sys.exit('The Column Number %d for file %s does not have a valid plane specified (should be %s)' %
                         (self.col_num, dataset_id, ', '.join(ColumnData.VALID_PLANES)))

    @staticmethod
    def generate_header_row():
        tmp_list = []
        for t in ColumnData.VALID_TYPES:
            if t in ColumnData.TYPE_HAS_COORDS:
                for p in ColumnData.VALID_PLANES:
                    tmp_list.append('%s %s' % (p, t))
            else:
                tmp_list.append(t)
        return tmp_list


class Joint:
    def __init__(self, joint_name):
        self.joint_name = joint_name
        self.joint_data = {}

    def update_joint(self, column_data, data, frame_num, framerate, y_fudge, low_point):
        if frame_num not in self.joint_data:
            self.joint_data.update({frame_num: JointData(frame_num, framerate)})
        self.joint_data[frame_num].add_joint_data(column_data, data, y_fudge, low_point)

    def __str__(self):
        output_str = 'Joint Name: %s' % self.joint_name
        for j in self.joint_data.values():
            output_str += str(j)
        return output_str

    def get_min(self, data_type, plane):
        tmp_list = []
        for v in self.joint_data.values():
            tmp_list.append(v.get(data_type, plane))
        return min(tmp_list)

    def get_max(self, data_type, plane):
        tmp_list = []
        for v in self.joint_data.values():
            tmp_list.append(v.get(data_type, plane))
        return max(tmp_list)

    def calc_averages(self):
        tmp_joint = JointData(1, 1)
        for data in self.joint_data.values():
            tmp_joint += data
        tmp_joint = tmp_joint / len(self.joint_data)
        return tmp_joint

    def get_average(self, data_type, plane):
        tmp_list = []
        for v in self.joint_data.values():
            tmp_list.append(v.get(data_type, plane))
        return np.mean(tmp_list)

    def generate_graph_data(self, x_data, y_data, phase_start, phase_end, graph_data):
        x_data_type = graph_data.x_axis.split(' ')
        y_data_type = graph_data.y_axis.split(' ')
        if graph_data.normalise:
            Joint.normalise_data(x_data, phase_start, phase_end)
        else:
            tmp_list = []
            for i in range(phase_start.frame_num, phase_end.frame_num+1):
                tmp_list.append(self.joint_data[i].get(x_data_type[1], x_data_type[0]))
            low_value = min(tmp_list)
            for v in tmp_list:
                x_data.append(v - low_value)
        for i in range(phase_start.frame_num, phase_end.frame_num+1):
            if len(y_data_type) == 2:
                y_data.append(self.joint_data[i].get(y_data_type[1], y_data_type[0]))
            else:
                y_data.append(self.joint_data[i].get(y_data_type[0], 'z'))

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

    def add_joint_data(self, column_data, data, y_fudge, low_point):
        if column_data.type in ColumnData.TYPE_HAS_COORDS:
            if column_data.type == 'position' and column_data.plane == 'y':
                self.data[column_data.type][column_data.plane] = float(data) +\
                                                                 (self.frame_num - 1) * y_fudge - low_point
            else:
                self.data[column_data.type][column_data.plane] = float(data)
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

    def __iadd__(self, other):
        if self.angle is None:
            self.angle = other.angle
        else:
            self.angle += other.angle
        for k1, v1 in self.data.items():
            for k2, v2 in v1.items():
                if self.data[k1][k2] is None:
                    self.data[k1][k2] = other.data[k1][k2]
                else:
                    self.data[k1][k2] += other.data[k1][k2]
        return self

    def __truediv__(self, other):
        if self.angle is not None:
            self.angle = self.angle / other
        for k1, v1 in self.data.items():
            for k2, v2 in v1.items():
                if self.data[k1][k2] is not None:
                    self.data[k1][k2] = self.data[k1][k2] / other
        return self

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
        self.normalise = (xml_data.get('normalise') == 'true')
        if xml_data.get("phase") is None:
            sys.exit("No Phase specified in Graph Definition for %s" % self.title)
        phase_key = xml_data.get("phase").replace(" ", "_")
        if phase_key in phases:
            self.phase = phases[phase_key]
        else:
            sys.exit("Unknown Phase specified in Graph titled %s" % self.title)
        x_axis = xml_data.find('x-axis')
        y_axis = xml_data.find('y-axis')
        if x_axis is None:
            sys.exit("There is no X Axis specified for the Graph titled %s" % self.title)
        if y_axis is None:
            sys.exit("There is no Y Axis specified for the Graph titled %s" % self.title)
        self.x_axis = x_axis.get("data")
        self.y_axis = y_axis.get("data")
        self.x_axis_title = x_axis.get("title")
        self.y_axis_title = y_axis.get("title")


class ExtractData:
    def __init__(self, xml_data, phases):
        self.name = xml_data.get("name")
        if self.name is None:
            sys.exit("A Defined Row for Data Extract does not have a name")
        try:
            self.value = xml_data.find('value').text
        except AttributeError:
            sys.exit("The Row %s does not have a Value specified" % self.name)
        self.joints = []
        for j in xml_data.findall('joint'):
            self.joints.append(j.text)
        if len(self.joints) == 0:
            sys.exit('The Row %s does not  have any Joints specified' % self.name)
        try:
            self.point = xml_data.find('point').text
        except AttributeError:
            self.point = None
        try:
            self.phase = xml_data.find('phase').text
        except AttributeError:
            self.phase = None
        if self.value == 'position':
            try:
                self.plane = xml_data.find('plane').text
            except AttributeError:
                sys.exit('The Row %s does not have a plane specified' % self.name)
            if self.plane not in ColumnData.VALID_PLANES:
                sys.exit('The Row %s does not have a valid plane (should be position or angle)' % self.name)
        else:
            self.plane = None
        if self.point is not None and self.phase is not None:
            sys.exit("The Defined Row %s has a Point and a Phase" % self.name)
        if self.point is None and self.phase is None:
            sys.exit("The Defined Row %s does not have a Point or Phase defined" % self.name)
        if self.value == 'position' and self.point is not None and len(self.joints) != 2:
            sys.exit("The Defined Row %s has incorrect data" % self.name)
        if self.phase is not None and len(self.joints) != 1:
            sys.exit("The Defined Row %s has incorrect data" % self.name)
        if self.phase is not None and self.phase not in phases:
            sys.exit("The Defined Row %s has an invalid phase (%s)" % (self.name, self.phase))

    def do_extract(self, csv_writer, samples, phases):
        output_row = [self.name]
        for s in samples.values():
            v = []
            if len(self.joints) == 2:
                for j in self.joints:
                    v.append(s.joints[j].joint_data[s.key_points[self.point].frame_num].data[self.value][self.plane])
            else:
                if self.value == 'position':
                    for p in [phases[self.phase].start, phases[self.phase].end]:
                        v.append(s.joints[self.joints[0]].joint_data[s.key_points[p].frame_num].
                                 data[self.value][self.plane])
                else:
                    v.append(s.joints[self.joints[0]].joint_data[s.key_points[self.point].frame_num].angle)
            if len(v) == 2:
                output_row.append(abs(v[0] - v[1]))
            else:
                output_row.append(v[0])
        csv_writer.writerow(output_row)


class FileInfo:
    def __init__(self, xml_data, sample_id):
        self.filename = xml_data.get("filename")
        self.framerate = xml_data.get("framerate")
        self.skip_count = xml_data.get("skip")
        self.data_type = xml_data.get("type")
        self.y_fudge = xml_data.get("y-fudge")
        self.zero_point = xml_data.get("zero-point")
        if self.filename is None:
            sys.exit("A Configuration or Data file for Sample ID %d does not have a filename specified" % sample_id)
        if not os.path.isfile(self.filename):
            sys.exit("A Configuration or Data File for Sample ID %d has a file (%s) that does not exit" %
                     (sample_id, self.filename))
        if self.framerate is None:
            sys.exit("A Configuration or Data File for Sample ID %d does not have a framerate specified" % sample_id)
        else:
            try:
                self.framerate = int(self.framerate)
            except ValueError:
                sys.exit("A Configuration or Data File for Sample ID %d does not have a valid framerate" % sample_id)
        if self.skip_count is None:
            self.skip_count = 0
        else:
            try:
                self.skip_count = int(self.skip_count)
            except ValueError:
                sys.exit("A Configuration or Data File for Sample ID %d has a non numeric framerate" % sample_id)
        if self.y_fudge is None:
            self.y_fudge = 0
        else:
            try:
                self.y_fudge = float(self.y_fudge)
            except ValueError:
                sys.exit("A Configuration or Data File for Sample ID %d has a non numeric y-fudge" % sample_id)


class Segment:
    def __init__(self, xml_data):
        self.name = xml_data.get('name')
        self.superior = xml_data.get('superior')
        self.inferior = xml_data.get('inferior')

    def get_length(self, sample, frame_num):
        superior_coords = [sample.joints[self.superior].joint_data[frame_num].get('position', 'x'),
                           sample.joints[self.superior].joint_data[frame_num].get('position', 'y')]
        inferior_coords = [sample.joints[self.inferior].joint_data[frame_num].get('position', 'x'),
                           sample.joints[self.inferior].joint_data[frame_num].get('position', 'y')]
        return math.sqrt((superior_coords[0] - inferior_coords[0])**2 + (superior_coords[1] - inferior_coords[1])**2)


def main():
    parser = OptionParser()
    parser.add_option('-x', '--xml_file', action='store', type='string', dest='xml_file')
    (opt, arg) = parser.parse_args()

    if opt.xml_file is None:
        parser.error('Please supply an xml file as an argument')
    if not os.path.isfile(opt.xml_file):
        parser.error('The Supplied XML file does not exist or is not a file')

    ds = KinematicData(opt.xml_file)
    ds.process_extract_data()
    ds.create_graphs()
    ds.get_segment_lengths()
    ds.raw_export_data()


if __name__ == '__main__':
    main()
