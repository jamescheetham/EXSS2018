from xml.etree import ElementTree as Et
import numpy as np
import sys
import csv
import os
import matplotlib.pyplot as plt
import math
from optparse import OptionParser

"""
KinematicData contains a dict of samples keyed by the sample id
A Sample contains a dict of joints keyed by the joint name
A Joint contains a dict of JointData keyed by the frame number
A JointData contains multidimensional dict keyed by the data type (position or velocity) and a plane (x or y) as well
as angle data (if relevant).

Copyright James Cheetham 2018
Version: 1.0
Last Modified: 20180509

"""


class KinematicData:
    """
    The Parent Class of the entire program. It takes the XML File with the relevant information in to the constructor
    which is then processed.
    It stores a Dict of Samples which is keyed by the sample ID.
    It has a Dict of Phases, as well as the information to graph, and to extract from the samples.
    """
    def __init__(self, xml_config_file):
        """
        Constructor for the Kinematic Data Class, takes an XML File as an argument which is then parsed
        for the relevant data. After this has been called, it should have read in and normalised all Samples
        :param xml_config_file:
        """
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
        """
        Reads the Config File. It calls relevant functions to populate the class variables.
        It will also have the samples populated from their files.
        :return:
        """
        self.xml = Et.parse(self.xml_config_file)
        self.xml_root = self.xml.getroot()
        definitions = self.xml_root.find('definitions')
        analysis = self.xml_root.find('analysis')
        self.generate_phases(definitions.find('phases'))
        self.generate_graph_data(analysis.find('graphs'))
        self.generate_extract_data(analysis.find('data'))
        self.generate_segment_data(definitions.find('segments'))

        # Loops through the dataset entries in the XML Data to create a new Sample for each of them
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
        """
        Reads the Phase Information from the XML File. Each phase requires a name, start and end.
        :param phases: XML Data
        :return: None
        """
        for p in phases.findall('phase'):
            name = p.get('name')
            start = p.get('start')
            end = p.get('end')

            # Checks that there is sufficient information to define a Phase
            if name is None or start is None or end is None:
                sys.exit('A Phase is defined without a Name, Start Point or End Point')

            self.phases.update({name.replace(' ', '_'): Phase(start, end)})

    def generate_graph_data(self, analysis_data):
        """
        Reads the Graph Data from the XML File
        :param analysis_data: XML Data
        :return:
        """
        for g in analysis_data.findall('graph'):
            self.graphs.append(Graph(g, self.phases))

    def generate_extract_data(self, analysis_data):
        """
        Reads the Information on which data to extract from the Samples
        :param analysis_data: XML Data
        :return: None
        """
        self.extract_data_filename = analysis_data.get('file')
        if self.extract_data_filename is None:
            sys.exit("The Analysis Data does not have a File Name specified")

        for d in analysis_data.findall('row'):
            self.extract_data.append(ExtractData(d, self.phases))

    def generate_segment_data(self, xml_data):
        """
        Extracts the Segment Data from the XML File
        :param xml_data: XML Data
        :return: None
        """
        for s in xml_data.findall('segment'):
            self.segments.append(Segment(s))

    def create_graphs(self):
        """
        Creates the Graphs from the Sample Data as per the information in the Graph variable
        :return: None
        """
        for g in self.graphs:
            legend_titles = []
            fig = plt.figure(figsize=(10, 10))
            plot = fig.add_subplot(1, 1, 1)
            plot.set_title(g.title)
            plot.set_xlabel(g.x_axis_title)
            plot.set_ylabel(g.y_axis_title)
            x_axis_min = None
            x_axis_max = None
            x_axis_labels = {}
            # Loops through the Samples to have the information added to the Graph
            i=0
            for k, v in self.samples.items():
                legend_titles.append(v.name)
                tmp_min, tmp_max = v.graph(g, plot, Graph.GRAPH_MARKERS[i], Graph.GRAPH_LINES[i], x_axis_labels)
                if x_axis_min is None or tmp_min < x_axis_min:
                    x_axis_min = tmp_min
                if x_axis_max is None or tmp_max > x_axis_max:
                    x_axis_max = tmp_max
                i += 1
                # Need to think about the X Axis to mark the key points. The sample has information on the keypoints, so I would 
                # have to loop through them and record the frame numbers, which would then have to be normalised like the data.
                # This would then have to be averaged between the samples.
            plot.legend(legend_titles)
            # noinspection PyTypeChecker
            plot.set_xlim([x_axis_min, x_axis_max])
            if len(x_axis_labels) > 0:
                x_ticks = []
                x_labels = []
                for k, v in x_axis_labels.items():
                    x_labels.append(k)
                    tick_sum = 0
                    for i in v:
                        tick_sum += i
                    x_ticks.append(tick_sum/len(v))
                plot.set_xticks(x_ticks)
                plot.set_xticklabels(x_labels)
            fig.savefig(g.filename)

    def process_extract_data(self):
        """
        Pulls the information from the Samples to export to the relevant file
        :return: None
        """
        with open(self.extract_data_filename, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = ['title']
            for s in self.samples.values():
                header.append('%d - %s - %s' % (s.sample_id, s.sample_type, s.name))
            csv_writer.writerow(header)
            for e in self.extract_data:
                e.do_extract(csv_writer, self.samples, self.phases)

    def get_segment_lengths(self):
        """
        Loops through the Samples and calls a function to write the segment lengths to a file.
        :return: None
        """
        for s in self.samples.values():
            s.write_segment_lengths(self.segments)

    def raw_export_data(self):
        """
        Exports the Sample Data to a different file.
        :return:
        """
        for s in self.samples.values():
            s.raw_export_data()


class Sample:
    """
    Holds the information relevant for an individual sample.
    The main data is the joints Dict, which is keyed by the Joint Name, and each entry is a JointData
    key_points is a Dict keyed by the Point Name, and the value is the frame number where that event occurs
    columns is a Dict keyed by the Column Type (which will need to match the file type), and each entry is a list
    containing the column data.
    The Calibration Data is a list of the Calibration Data from the files. It's possible to supply multiple calibration
    files. Each List is a Dict of JointData
    calibration_zeros contains the average of the Calibration Data
    """
    def __init__(self, sample_type, data_set_info, sample_id, name):
        """
        Constructor for the Sample. Initialises all variables and calls process to pull all relevant data.
        :param sample_type: String: The sample type
        :param data_set_info: XML Data
        :param sample_id: int: The id of the sample
        :param name: String: The name of the sample
        """
        self.sample_id = sample_id
        self.sample_type = sample_type
        self.name = name
        self.joints = {}
        self.key_points = {}
        self.columns = {}
        self.calibration_data = []
        self.calibration_zeros = {}
        self.angular_columns = None
        self.frame_count = 0
        self.process(data_set_info)

    def process(self, data_set_info):
        """
        Call relevant functions to read the column data, the calibration data, calculate the calibration zeros
        and read the sample data from the specified files
        :param data_set_info: XML Data
        :return: None
        """
        self.generate_column_data(data_set_info.find('columns'))
        print("Generated Column Data")
        self.get_calibration_data(data_set_info.findall('calibration'))
        print("Pulled Calibration Data")
        xml_normalisation = data_set_info.find('normalisation')
        low_point = self.calc_calibration_zeros(xml_normalisation)
        self.angular_columns = xml_normalisation.findall('angle_correction')
        datafiles = data_set_info.find('datafile')

        # Loops through the details entries in the XML File to extract the data for the Sample
        for details in datafiles.findall('details'):
            file_info = FileInfo(details, self.sample_id)
            self.process_datafile(file_info, low_point)
            print("Processed DataFile %s" % file_info.filename)

        kp = data_set_info.find('keypoints')

        # Loops through the Point Information to calibrate the Key Points
        for p in kp.findall('point'):
            self.key_points.update({p.get('name'): KeyPoint(p)})

    def get_calibration_data(self, xml_calibration):
        """
        Processes the Calibration Information for the Sample.
        :param xml_calibration: XML Data
        :return: None
        """
        # There can be multiple Calibration Files for a single Sample, so they all have to be processed
        for cali in xml_calibration:
            tmp_joint_data = {}

            # Read each file specified in the details of the XML File
            for c in cali.findall('details'):
                file_info = FileInfo(c, self.sample_id)
                with open(file_info.filename, 'r') as f:
                    # The Default export from Skill Spector uses Tab as a delimiter
                    csv_reader = csv.reader(f, delimiter='\t', quotechar='"')

                    # Skips a set number of rows (Skill Spector file header)
                    for _ in range(file_info.skip_count):
                        next(csv_reader)

                    self.read_file(csv_reader, file_info, tmp_joint_data)

            self.calibration_data.append(tmp_joint_data)

    def calc_calibration_zeros(self, xml_data):
        """
        Averages the Calibration Data to calculate the zeros or averages for the natural position
        :param xml_data: XML Data
        :return: None
        """
        averages = []
        # Averages the values for each entry in Calibration Data
        for c in self.calibration_data:
            tmp_data = {}
            # Loop through the Joint Data in the Calibration Data, calculate their averages and append it to
            # the averages list
            for k, v in c.items():
                tmp_data.update({k: v.calc_averages()})

            averages.append(tmp_data)

        # If there was more than one file, then average them between each other
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
        # If there is normalisation data in the XML File, extract the joint name for the Low Point and return it
        if xml_data is not None:
            low_joint_xml = xml_data.find('low_point')
            if low_joint_xml is not None:
                low_joint_name = low_joint_xml.get('joint')
                # Ensure that the low joint exists and is a valid joint name
                if low_joint_name is None or low_joint_name not in self.calibration_zeros:
                    sys.exit('The Calibration Data specifies a Joint for low_point for Sample %d \
                    that is not defined (%s)'
                             % (self.sample_id, low_joint_name))

                return self.calibration_zeros[low_joint_name].get('position', 'y')

        return 0

    def read_file(self, csv_reader, file_info, joint_data, low_point=0):
        """
        Read a Skill Spector CSV File
        :param csv_reader: The csv reader object to process
        :param file_info: A File Info object containing the information about reading the file
        :param joint_data: The Joint Data Dict to populate
        :param low_point: The low point to normalise against
        :return: The number of frames in the file
        """
        line_count = 1
        for line in csv_reader:
            # Populate the Joint Data as per the Column Information
            for c in self.columns[file_info.data_type]:
                # If the Joint doesn't current exist in the joint_data, add it.
                if c.joint not in joint_data:
                    joint_data.update({c.joint: Joint(c.joint)})
                try:
                    joint_data[c.joint].update_joint(c, line[c.col_num - 1], line_count,
                                                     file_info.framerate, file_info.y_fudge, low_point)
                except ValueError:
                    # If the file line contains non numeric data, the program will throw this error
                    sys.exit("There was an error processing the file %s in column %d for frame number %d" %
                             (file_info.filename, c.col_num, line_count + 1))
            line_count += 1

        return line_count - 1

    def process_datafile(self, file_info, low_point):
        """
        Reads a Data File and processes it into the joints class variable
        :param file_info: A FileInfo Object containing the file information
        :param low_point: The Low Point to normalise against
        :return:
        """
        with open(file_info.filename, 'r') as f:
            # Skill Spector files are tab delimited
            csv_reader = csv.reader(f, delimiter='\t', quotechar='"')
            # Skip the Header Rows
            for _ in range(file_info.skip_count):
                next(csv_reader)

            self.frame_count = self.read_file(csv_reader, file_info, self.joints, low_point)

        for j in self.joints.values():
            j.normalise_angles(self.calibration_zeros, self.angular_columns)

    def generate_column_data(self, xml_columns):
        """
        Read the Column Information from the XML File and populate the class variable columns
        :param xml_columns: XML Data
        :return:
        """
        # Loop through the different column types in the XML Data
        for column_type in xml_columns:
            self.columns.update({column_type.tag: []})
            cols = column_type.findall('col')
            # Create a list to a set length
            self.columns[column_type.tag] = [None] * len(cols)

            for c_data in cols:
                tmp_col = ColumnData(c_data, self.sample_id)
                # Ensure that Column Numbers aren't duplicated
                if self.columns[column_type.tag][tmp_col.col_num - 1] is not None:
                    sys.exit('The Column Number %d for file %s is duplicated' % (tmp_col.col_num, self.name))
                try:
                    # Ensure that the column number isn't too high
                    self.columns[column_type.tag][tmp_col.col_num - 1] = tmp_col
                except IndexError:
                    sys.exit('The Column Number %d for file %s is too high' % (tmp_col.col_num, self.name))

            # Ensure that all of the column data is supplied
            if None in self.columns[column_type.tag]:
                sys.exit('There was an error processing Column Data for the file %s' % self.name)

    def graph(self, graph_data, fig, marker, line_type, x_axis_labels):
        """
        Graph the Sample onto a figure. It will always plot as a Black, using the Marker and Line Type provided
        :param graph_data: The GraphData Object
        :param fig: The PyPlot Figure
        :marker: The Marker Type to use from MatPlotLib
        :line_type: The Line Type to use from MatPlotLib
        :return: The Max and Minimum values for the X-Axis
        """
        x_data = []
        y_data = []
        if graph_data.joint in self.joints and graph_data.graph_type == 'line':
            phase_start = graph_data.phase.start
            phase_end = graph_data.phase.end
            # Check that the phase start and phase end are valid for this data
            if phase_start not in self.key_points or phase_end not in self.key_points:
                sys.exit("Key Point %s or %s not defined for Sample ID %d"
                         % (phase_start, phase_end, self.sample_id))

            self.joints[graph_data.joint].generate_graph_data(x_data, y_data, self.key_points[phase_start],
                                                              self.key_points[phase_end], graph_data)
            if graph_data.normalise:
                self.generate_x_axis_label_data(self.key_points[phase_start].frame_num,
                                                self.key_points[phase_end].frame_num, x_axis_labels)
        elif graph_data.graph_type == 'xy_plot':
            x_adjustment = self.joints[graph_data.normalise_joint].joint_data[
                self.key_points[graph_data.point].frame_num].get('position', 'x')
            for j in graph_data.joint_order:
                if graph_data.point not in self.key_points:
                    sys.exit("Key Point %s not specified for Sample ID %d" % (graph_data.point, self.sample_id))
                if j not in self.joints:
                    sys.exit("Joint %s specified for Graph %s does not exist in Sample %d" %
                             (j, graph_data.title, self.sample_id))
                self.joints[j].generate_xy_graph_data(x_data, y_data,
                                                      self.key_points[graph_data.point].frame_num, x_adjustment)
        if len(x_data) > 0 and len(y_data) > 0:
            if graph_data.graph_type == 'xy_plot':
                fig.plot(x_data, y_data, marker=marker, markersize='6', linestyle=line_type, color='k')
            else:
                fig.plot(x_data, y_data, linestyle=line_type, color='k')
            return min(x_data), max(x_data)
        return 0, 0
    
    def generate_x_axis_label_data(self, phase_start, phase_end, x_axis_labels):
        for kp in self.key_points.values():
            if phase_start <= kp.frame_num <= phase_end:
                if kp.name not in x_axis_labels:
                    x_axis_labels.update({kp.name: []})
                x_axis_labels[kp.name].append((kp.frame_num - phase_start)/(phase_end - phase_start)*100)

    def write_segment_lengths(self, segments):
        """
        Write the lengths of each segment for each frame to a file
        :param segments: The list of Segments
        :return: None
        """
        filename = 'segment_lengths_%s.csv' % self.name

        with open(filename, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['frame number'] + [x.name for x in segments])
            # Loop through each Frame and write the segment lengths
            for i in range(1, self.frame_count + 1):
                tmp_array = [i]
                for s in segments:
                    tmp_array.append(s.get_length(self, i))
                csv_writer.writerow(tmp_array)

    def raw_export_data(self):
        """
        Export the Joint Data as read into the system
        :return: None
        """
        filename = 'data_export_%s.csv' % self.name
        with open(filename, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            base_header = ColumnData.generate_header_row()
            header = []
            # Generate the Header Row
            for j in self.joints.keys():
                for h in base_header:
                    header.append('%s %s' % (j, h))
            csv_writer.writerow(['framenumber'] + header)
            # Write the Calibration zeros to the file
            output_list = ['calibration']
            # The Header also contains the information to export in each column
            for v in header:
                tmp = v.split(' ')
                if len(tmp) == 2:
                    # Header is <joint> angle
                    output_list.append(self.calibration_zeros[tmp[0]].angle)
                else:
                    # Header is <joint> <plane> <data type>
                    output_list.append(self.calibration_zeros[tmp[0]].data[tmp[2]][tmp[1]])

            csv_writer.writerow(output_list)
            for i in range(1, self.frame_count + 1):
                output_list = [i]
                for v in header:
                    tmp = v.split(' ')
                    if len(tmp) == 2:
                        # Header is <joint> angle
                        output_list.append(self.joints[tmp[0]].joint_data[i].angle)
                    else:
                        # Header is <joint> <plane> <data type>
                        output_list.append(self.joints[tmp[0]].joint_data[i].data[tmp[2]][tmp[1]])

                csv_writer.writerow(output_list)


class ColumnData:
    """
    Stores the Column Data for the Samples
    """
    VALID_TYPES = ['position', 'angle', 'velocity']  # Valid Column Types
    TYPE_HAS_COORDS = ['position', 'velocity']       # The Column Types that will have planes associated with them
    VALID_PLANES = ['x', 'y']                        # The planes that are associated with the data types

    def __init__(self, col_data, dataset_id):
        """
        Pulls the information from the XML Data to create a ColumnData Object.
        Each Column must have a type, joint, and a plane if the type should have it.
        :param col_data: The XML Data holding the Column Information
        :param dataset_id: The ID of the sample
        """
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
        """
        Generates the Header Row based on the Static Variables
        :return: A List containing strings to represent the values for a Header Row in a CSV Export File
        """
        tmp_list = []
        for t in ColumnData.VALID_TYPES:
            if t in ColumnData.TYPE_HAS_COORDS:
                for p in ColumnData.VALID_PLANES:
                    tmp_list.append('%s %s' % (p, t))
            else:
                tmp_list.append(t)
        return tmp_list


class Joint:
    """
    Contains the information on a Joint. It consists of a Name, and a Dict (indexed by the frame number) of the Joint
    Data
    """
    def __init__(self, joint_name):
        """
        Creates the Joint Object
        :param joint_name: The name of the Joint
        """
        self.joint_name = joint_name
        self.joint_data = {}

    def update_joint(self, column_data, data, frame_num, framerate, y_fudge, low_point):
        """
        Adds information into the Joint Data Object
        :param column_data: The Column Data associated with the File
        :param data: The value read from the file
        :param frame_num: The frame number to associate with this data point
        :param framerate: The Frame Rate that the filming occurred at
        :param y_fudge: The fudge factor to apply to account for filming that wasn't on a flat surface
        :param low_point: The minimum height point to normalise to
        :return:
        """
        if frame_num not in self.joint_data:
            self.joint_data.update({frame_num: JointData(frame_num, framerate)})
        self.joint_data[frame_num].add_joint_data(column_data, data, y_fudge, low_point)

    def normalise_angles(self, calibration_zeros, xml_data):
        """
        Updates the angle data to set them as a value of flexion or extension from the calibration angles.
        It uses the multipler value from the XML file to determine if the direction for flexion needs to be adjusted
        :param calibration_zeros: A JointData Dict that contains the zero values from calibration data
        :param xml_data: The XML data that contains the Joint Correction value multipliers
        :return: None
        """
        multipliers = {}
        for x in xml_data:
            joint_name = x.get('joint')
            try:
                m = int(x.get('multiplier'))
            except ValueError:
                sys.exit('The Multiplier Value for Joint Angle Normalisation for %s is not a valid value' % joint_name)
            multipliers.update({joint_name: m})
        for j in self.joint_data.values():
            if self.joint_name in multipliers:
                j.normalise_angles(calibration_zeros[self.joint_name].get('angle', 'z'), multipliers[self.joint_name])
            else:
                j.normalise_angles(calibration_zeros[self.joint_name].get('angle', 'z'))

    def __str__(self):
        """
        Overrides the String Object to present a nice output of the Joint
        :return: String
        """
        output_str = 'Joint Name: %s' % self.joint_name
        for j in self.joint_data.values():
            output_str += str(j)

        return output_str

    def get_min(self, data_type, plane):
        """
        Returns the Minimum Value from within the data_type and plane
        :param data_type: The Data Type (from ColumnData) to get the minimum of
        :param plane: The Plane to extract the Data from
        :return: The minimum value from the JointData
        """
        tmp_list = []
        for v in self.joint_data.values():
            tmp_list.append(v.get(data_type, plane))

        return min(tmp_list)

    def get_max(self, data_type, plane):
        """
        Returns the Maximum Value from within the data_type and plane
        :param data_type: The Data Type (from ColumnData) to get the maximum of
        :param plane: The Plane to extract the Data from
        :return: The minimum value from the JointData
        """
        tmp_list = []
        for v in self.joint_data.values():
            tmp_list.append(v.get(data_type, plane))

        return max(tmp_list)

    def calc_averages(self):
        """
        Calculates the averages of each Data Point within the JointData
        :return: JointData Object containing the averages
        """
        # Create a Dummy JointData with dummy frame number and framerate
        tmp_joint = JointData(1, 1)
        for data in self.joint_data.values():
            tmp_joint += data

        tmp_joint = tmp_joint / len(self.joint_data)
        return tmp_joint

    def get_average(self, data_type, plane):
        """
        Returns the Average Value from within the data_type and plane
        :param data_type: The Data Type (from ColumnData) to get the average of
        :param plane: The Plane to extract the Data from
        :return: The average value from the JointData
        """
        tmp_list = []
        for v in self.joint_data.values():
            tmp_list.append(v.get(data_type, plane))

        return np.mean(tmp_list)

    def generate_graph_data(self, x_data, y_data, phase_start, phase_end, graph_data):
        """
        Generates the Graph Data for a Graph Object
        :param x_data: String, containing the data to extract
        :param y_data: String, containing the data to extract
        :param phase_start: String, The Key Point to start at
        :param phase_end: String, The Key Point to end at
        :param graph_data: Graph Object
        :return: None
        """
        x_data_type = graph_data.x_axis.split(' ')
        y_data_type = graph_data.y_axis.split(' ')
        if graph_data.normalise:
            Joint.normalise_data(x_data, phase_start, phase_end)
        else:
            tmp_list = []
            # Iterate through the frames from phase_start to phase_end
            for i in range(phase_start.frame_num, phase_end.frame_num+1):
                tmp_list.append(self.joint_data[i].get(x_data_type[1], x_data_type[0]))

            low_value = min(tmp_list)
            # Normalise the data to a left zero point
            for v in tmp_list:
                x_data.append(v - low_value)

        # Iterate through the frames from phase_start to phase_end
        for i in range(phase_start.frame_num, phase_end.frame_num+1):
            # Either extract the <plane> <position|velocity> or the angle
            if len(y_data_type) == 2:
                y_data.append(self.joint_data[i].get(y_data_type[1], y_data_type[0]))
            else:
                y_data.append(self.joint_data[i].get(y_data_type[0], 'z'))

    def generate_xy_graph_data(self, x_data, y_data, frame_num, x_adjustment):
        """
        Generates XY Plot data
        :param x_data: The array to contain the X Points to plot
        :param y_data: The array to contain the Y Points to plot
        :param frame_num: The frame number of the data to plot
        :param x_adjustment: The value to adjust the X value by so that the plot is stacked
        :return:
        """
        x_data.append(self.joint_data[frame_num].get('position', 'x') - x_adjustment)
        y_data.append(self.joint_data[frame_num].get('position', 'y'))

    @staticmethod
    def normalise_data(x_data, phase_start, phase_end):
        """
        Normalises the Data to be over 100% of the Phase
        :param x_data: List, contains the values to normalise
        :param phase_start: Phase Object, where the start point is
        :param phase_end: Phase Object, where the end point is
        :return:
        """
        phase_length = phase_end.frame_num - phase_start.frame_num
        for i in range(phase_start.frame_num, phase_end.frame_num+1):
            x_data.append((i - phase_start.frame_num)/phase_length*100)


class JointData:
    """
    Contains information about a Joint at a set point in time. The information it contains is set by the ColumnData
    Static variables
    """
    def __init__(self, frame_num, framerate):
        """
        Creates a JointData Object
        :param frame_num: The frame number
        :param framerate: The frame rate
        """
        self.frame_num = frame_num
        self.framerate = framerate
        self.timestamp = frame_num * 1 / framerate
        # Creates a Dict of Dicts based on the ColumnData Static Variables
        self.data = {i: {j: None for j in ColumnData.VALID_PLANES} for i in ColumnData.TYPE_HAS_COORDS}
        self.angle = None

    def add_joint_data(self, column_data, data, y_fudge, low_point):
        """
        Adds new Data to an existing Joint
        :param column_data: ColumnData Object
        :param data: The data value read in from the file
        :param y_fudge: The Fudge Factor to account for a non-level filming scene
        :param low_point: The Low Point to normalise against
        :return:
        """
        if column_data.type in ColumnData.TYPE_HAS_COORDS:
            # Only correct the Y Position data
            if column_data.type == 'position' and column_data.plane == 'y':
                # Add a fudge factor depending on how far through the scene the framenumber is
                self.data[column_data.type][column_data.plane] = float(data) +\
                                                                 (self.frame_num - 1) * y_fudge - low_point
            else:
                self.data[column_data.type][column_data.plane] = float(data)
        else:
            self.angle = float(data)

    def normalise_angles(self, calibration_zero, multiplier=1):
        """
        Normalises the angles for the Joint Data
        :param calibration_zero: The value that is considered "zero", i.e. where the joint is at 0 degrees flexion
        :param multiplier: The correction multiplier, if it is -1, then it indicates that the joint flexes backwards
        :return:
        """
        if self.angle is not None:
            self.angle = (self.angle - calibration_zero) * multiplier

    def get(self, data_type, plane):
        """
        Extract the data from the JointData
        :param data_type: String, the data type to extract
        :param plane: String, the plane to extract
        :return: Float, the value
        """
        if data_type in ColumnData.VALID_TYPES:
            if data_type in ColumnData.TYPE_HAS_COORDS:
                if plane in ColumnData.VALID_PLANES:
                    return self.data[data_type][plane]
            else:
                return self.angle
        return None

    def __iadd__(self, other):
        """
        Overloads the += operator for JointData
        :param other: The Other JointData Object to add
        :return: The JointData containing the sum of the two objects
        """
        tmp_jd = JointData(self.frame_num, self.framerate)
        # Adds the Angles
        if self.angle is None:
            tmp_jd.angle = other.angle
        else:
            tmp_jd.angle = self.angle + other.angle

        # Adds the Plane Information
        for k1, v1 in self.data.items():
            for k2, v2 in v1.items():
                if self.data[k1][k2] is None:
                    tmp_jd.data[k1][k2] = other.data[k1][k2]
                else:
                    tmp_jd.data[k1][k2] = self.data[k1][k2] + other.data[k1][k2]

        return tmp_jd

    def __truediv__(self, other):
        """
        Overloads the Division Operator, allows a Joint Data to be divided
        :param other: Float/Int, the value to divide by
        :return: The JointData divided by other
        """
        tmp_jd = JointData(1, 1)
        # Divides the Angle
        if self.angle is not None:
            tmp_jd.angle = self.angle / other

        # Divides the Plane Data
        for k1, v1 in self.data.items():
            for k2, v2 in v1.items():
                if self.data[k1][k2] is not None:
                    tmp_jd.data[k1][k2] = self.data[k1][k2] / other

        return tmp_jd

    def __str__(self):
        """
        Generates a well presented String for output
        :return: String
        """
        output_str = '\nTimestamp: %0.2f' % self.timestamp
        for k1, v1 in self.data.items():
            for k2, v2 in v1.items():
                if v2 is not None:
                    output_str += '\n%s %s: %0.3f' % (k2.upper(), k1.capitalize(), v2)

        if self.angle is not None:
            output_str += '\nAngle: %0.2f' % (0 if self.angle is None else self.angle)

        return output_str


class KeyPoint:
    """
    Holds the information about a Key Point
    """
    def __init__(self, point_data):
        """
        Extracts the Key Point information from the XML Data
        :param point_data: The XML Data about the Point. The name is a String and the framenum is an integer
        """
        self.name = point_data.get('name')
        try:
            self.frame_num = int(point_data.get('framenum'))
        except ValueError:
            sys.exit("The Frame Number for the Key Point %s is not a valid integer" % self.name)


class Phase:
    """
    Holds information about the Phase
    """
    def __init__(self, start, end):
        """
        :param start: The Start Point
        :param end: The End Point
        """
        self.start = start
        self.end = end


class Graph:
    """
    Graph Object that holds the information from the XML Data
    """
    GRAPH_MARKERS = ['o', 'v', '^', '>', '<']
    GRAPH_LINES = ['-', '--', '-.', ':']
    def __init__(self, xml_data, phases):
        """
        Extracts the information from the XML File for a Graph. Requires the specified phase to be be previously
        defined
        :param xml_data: XML Data
        :param phases: Phase Object list
        """
        self.graph_type = xml_data.get('type')
        self.filename = xml_data.get("file")
        self.title = xml_data.get("title")
        self.joint = xml_data.get("joint")
        self.point = xml_data.get("point")
        self.phase = None
        self.joint_order = None
        self.normalise = (xml_data.get('normalise') == 'true')
        self.normalise_joint = xml_data.get('normalise_joint')
        # The graph type can be either line or XY Plot
        if self.graph_type == 'line':
            if xml_data.get("phase") is None:
                sys.exit("No Phase specified in Graph Definition for %s" % self.title)
            else:
                phase_key = xml_data.get("phase").replace(" ", "_")
                if phase_key in phases:
                    self.phase = phases[phase_key]
                else:
                    sys.exit("Unknown Phase specified in Graph titled %s" % self.title)

        elif self.graph_type == 'xy_plot':
            self.joint_order = []
            for j in xml_data.findall('joint'):
                self.joint_order.append(j.text)

        # Error checking to ensure that there is correct data for a graph
        if self.graph_type == 'xy_plot' and self.point is None:
            sys.exit("No Point specified in Graph Definition for %s" % self.title)

        if self.graph_type == 'xy_plot' and self.normalise_joint is None:
            sys.exit("No Normalise Joint specified in Graph Definition for %s" % self.title)

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
    """
    Stores the information from the XML File relating to the values to extract from the Samples
    """
    def __init__(self, xml_data, phases):
        """
        Pulls the data from the XML File for an Extract of Data. There are two types of extracts.
        Two Joints - Requires two joints, a point, a plane and data type. This will determine the difference between the
        two joints at the specified point
        Single Joint - Requires a Phase, a point, a plane and data type. This will calculate the distance between
        the joint from the start to the end of the phase.
        :param xml_data: XML Data
        :param phases: Phase information
        """
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
        """
        Iterates through the samples and extracts the information. The information is then written to the
        CSV File.
        :param csv_writer: The CSV Writer Object
        :param samples: The Samples Objects
        :param phases: Phases List
        :return: None
        """
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
    """
    Holds the Information of a Skill Spector File
    """
    def __init__(self, xml_data, sample_id):
        """
        Pulls the relevant information and checks it's integrity from the XML Data. The XML Data must contain the
        filename (which must exist), framerate, type. Optional values are y-fudge and skip.
        :param xml_data: The XML Data
        :param sample_id: The sample id of the file
        """
        self.filename = xml_data.get("filename")
        self.framerate = xml_data.get("framerate")
        self.skip_count = xml_data.get("skip")
        self.data_type = xml_data.get("type")
        self.y_fudge = xml_data.get("y-fudge")
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
        if self.data_type is None:
            sys.exit("A Configuration or Data File for Sample ID %d does not have a type" % sample_id)
        if self.y_fudge is None:
            self.y_fudge = 0
        else:
            try:
                self.y_fudge = float(self.y_fudge)
            except ValueError:
                sys.exit("A Configuration or Data File for Sample ID %d has a non numeric y-fudge" % sample_id)


class Segment:
    """
    Holds the information fo a Segment. A Segment is defined by the superior and inferior joints
    """
    def __init__(self, xml_data):
        """
        Reads the name, superior and inferior joints for the Segment
        :param xml_data: XML Data
        """
        self.name = xml_data.get('name')
        self.superior = xml_data.get('superior')
        self.inferior = xml_data.get('inferior')

    def get_length(self, sample, frame_num):
        """
        Calculates the Length of the Segment for the given frame number from the Sample
        :param sample: The Sample to get the Segment from
        :param frame_num: The Framenumber to calculate the Segment for
        :return: The Length of the Segment
        """
        superior_coords = [sample.joints[self.superior].joint_data[frame_num].get('position', 'x'),
                           sample.joints[self.superior].joint_data[frame_num].get('position', 'y')]
        inferior_coords = [sample.joints[self.inferior].joint_data[frame_num].get('position', 'x'),
                           sample.joints[self.inferior].joint_data[frame_num].get('position', 'y')]
        return math.sqrt((superior_coords[0] - inferior_coords[0])**2 + (superior_coords[1] - inferior_coords[1])**2)


def main():
    """
    Takes a single argument of the xml file that contains all of the Kinematics Information
    :return:
    """
    parser = OptionParser()
    parser.add_option('-x', '--xml_file', action='store', type='string', dest='xml_file')
    (opt, arg) = parser.parse_args()

    if opt.xml_file is None:
        parser.error('Please supply an xml file as an argument')
    if not os.path.isfile(opt.xml_file):
        parser.error('The Supplied XML file does not exist or is not a file')

    # Create a KinematicData Ojbect, process the Extract Data, Graphs, Segment Lengths and export the data
    kinematic_data = KinematicData(opt.xml_file)
    kinematic_data.process_extract_data()
    kinematic_data.create_graphs()
    kinematic_data.get_segment_lengths()
    kinematic_data.raw_export_data()


if __name__ == '__main__':
    main()
