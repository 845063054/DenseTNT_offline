import os
from sqlite3 import Timestamp
import sys
import carla
import math
import csv
from yaml_utils import load_yaml
from carla_map_api import MapManager
import time
from collections import deque
import numpy as np

class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """
    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = self.get_speed(self._vehicle)

        print('Current speed = {}'.format(current_speed))

        acceleration = self._pid_control(target_speed, current_speed)
        
        if acceleration >= 0.0:
            throttle = min(acceleration, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(acceleration), 1.0)
        
        return throttle,brake

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt

    def get_speed(self,vehicle):
        vel = vehicle.get_velocity()

        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def set_weather(weather_settings):
    """
    Set CARLA weather params.
    """
    weather = carla.WeatherParameters(
        sun_altitude_angle=weather_settings['sun_altitude_angle'],
        cloudiness=weather_settings['cloudiness'],
        precipitation=weather_settings['precipitation'],
        precipitation_deposits=weather_settings['precipitation_deposits'],
        wind_intensity=weather_settings['wind_intensity'],
        fog_density=weather_settings['fog_density'],
        fog_distance=weather_settings['fog_distance'],
        fog_falloff=weather_settings['fog_falloff'],
        wetness=weather_settings['wetness']
    )
    return weather


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def run_step():
    try:
        config_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)),'carla_config.yaml')
        if not os.path.isfile(config_yaml):
            sys.exit("carla_config.yaml not found!")
        config_params = load_yaml(config_yaml)
        world_params = config_params['world']
        # load the carla client
        client = carla.Client('localhost', world_params["client_port"])
        client.set_timeout(world_params["timeout"])
        client.load_world(world_params["map_name"])
        world = client.get_world()
        carla_map = world.get_map()
        origin_settings = world.get_settings()
        blueprint_library = world.get_blueprint_library()
        #set the sync_mode
        settings = world.get_settings()     
        settings.fixed_delta_seconds = world_params['fixed_delta_seconds']
        settings.synchronous_mode = world_params['sync_mode']

        # set weather
        weather = set_weather(world_params["weather"])
        world.set_weather(weather)

        # set the traffic manager sync
        tarffic_manager_params = config_params['traffic_manager']
        traffic_manager_ = client.get_trafficmanager()
        traffic_manager_.set_synchronous_mode(tarffic_manager_params['sync_mode'])
        world.apply_settings(settings)
        world.tick()

        ego_params = config_params['ego_vehicle']
        # vehicle_transform = carla.Transform(carla.Location(x=ego_params['spawn_position'][0],
        #                         y=ego_params['spawn_position'][1],
        #                         z=ego_params['spawn_position'][2]),
        #                     carla.Rotation(
        #                         pitch=ego_params['spawn_position'][3],
        #                         yaw=ego_params['spawn_position'][4],
        #                         roll=ego_params['spawn_position'][5]))

        all_default_spawn = world.get_map().get_spawn_points()
        vehicle_transform = all_default_spawn[0]
        print(vehicle_transform)
        
        vehicle_bp = blueprint_library.find(ego_params['brand'])
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
        vehicle.set_simulate_physics(ego_params['simulate_physics'])
        vehicle.set_autopilot(ego_params['autopilot'])
        traffic_manager_.ignore_signs_percentage(vehicle, ego_params['ignore_signs_percentage'])

        longcontrol_pid = PIDLongitudinalController(vehicle)
        # target speed km/h
        control = carla.VehicleControl()
        # set the traffic light is Green
        for actor in world.get_actors():
            if actor.type_id == 'traffic.traffic_light':
                actor.freeze(True)
                actor.set_state(carla.TrafficLightState.Green)

        trajectory_number = 0
        ego_trajectory_path = '/home/user/Database/Dense_carla_dataset/val/' + world_params["map_name"] 
        if not os.path.isdir(ego_trajectory_path):
            os.makedirs(ego_trajectory_path)

        map_manager = MapManager(world_params["map_name"])

        trajectory_number_str = str(trajectory_number).zfill(4)
        ego_trajectory_path_csv = ego_trajectory_path + '/' + trajectory_number_str +'.csv'
        csvfile = open(ego_trajectory_path_csv,'w',newline='')
        writer = csv.writer(csvfile)
        item = ['TIMESTAMP','TRACK_ID','OBJECT_TYPE','X','Y','CITY_NAME']
        writer.writerow(item)
        init_time = time.time()

        count = 0
        number = 0
        number_ = 0
        while True:
            world.tick()
            # agent._update_information(vehicle)
            ego_pose = vehicle.get_transform()
            x = ego_pose.location.x
            y = ego_pose.location.y
            count += 1
            if count > 12:
                if number > 49:
                    number = 0
                    csvfile.close()
                    trajectory_number += 1
                    trajectory_number_str = str(trajectory_number).zfill(4)
                    ego_trajectory_path_csv = ego_trajectory_path + '/' + trajectory_number_str +'.csv'
                    csvfile = open(ego_trajectory_path_csv,'w',newline='')
                    writer = csv.writer(csvfile)
                    item = ['TIMESTAMP','TRACK_ID','OBJECT_TYPE','X','Y','CITY_NAME']
                    writer.writerow(item)
                    init_time = time.time()
                time_current = time.time() - init_time
                value = [time_current,trajectory_number_str,'AGENT',x,y,world_params["map_name"]]
                writer.writerow(value)
                number += 1
                count = 0
            # # test
            # map_manager.update_state(ego_pose)
            # lane_ids = map_manager.get_lane_ids_in_xy_bbox(query_search_range_manhattan = 50)
            # print(lane_ids)
            # local_lane_centerlines = [map_manager.get_lane_segment_centerline(lane_id) for lane_id in lane_ids]
            # lane_ids111 = map_manager.find_local_lane_centerlines(query_search_range_manhattan = 50)
            # vis_lanes = [map_manager.get_lane_segment_polygon(lane_id)[:, :2] for lane_id in lane_ids]
            # t = []
            # angle = 0
            # for each in vis_lanes:
            #     for point in each:
            #         point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)
            #     num = len(each) // 2
            #     t.append(each[:num].copy())
            #     t.append(each[num:num * 2].copy())
            # vis_lanes = t

            # lane_segment = [map_manager.city_lane_centerlines_dict(lane_id) for lane_id in lane_ids]
            # print(lane_segment[0].get('has_traffic_control'))

            # set spectator
            spectator_location = ego_pose.location + carla.Location(z = 300)
            spectator_rotation = carla.Rotation(pitch = -90)
            spectator_transform = carla.Transform(spectator_location, spectator_rotation)
            spectator = world.get_spectator()
            spectator.set_transform(spectator_transform)
            number_ += 1
            if number_ > 1:
                control_la = vehicle.get_control()
                throttle,brake = longcontrol_pid.run_step(15)
                control_la.throttle = throttle
                # print(control_la)
                vehicle.apply_control(control_la)
                number_ = 2

    finally:
        csvfile.close()
        world.apply_settings(origin_settings)
        vehicle.destroy()
        traffic_manager_.set_synchronous_mode(False)
        # for sensor in sensor_list:
        #     sensor.destroy()


    # # set the traffic light is Green
    # for actor in world.get_actors():
    #     if actor.type_id == 'traffic.traffic_light':
    #         actor.freeze(True)
    #         actor.set_state(carla.TrafficLightState.Green)

if __name__ == "__main__":
    try:
        run_step()
    except KeyboardInterrupt:
        print(' - Exited by user.')

