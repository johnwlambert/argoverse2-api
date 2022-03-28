# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Map change detection evaluation script."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import av2.utils.io as io_utils
import click
import numpy as np
from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.map import EgoViewMapRenderer
from shapely.geometry import Polygon, Point
from shapely.geometry import LinearRing, Point, Polygon


DUAL_CATEGORY_DICT = {
    "delete_crosswalk": "insert_crosswalk",
    "insert_crosswalk": "delete_crosswalk",
    "lane_geometry_change": "lane_geometry_change"  # TODO: add fine-grained
    # 'change_lane_marking_color':'change_lane_marking_color'
    # 'delete_lane_marking': 4,
    # 'add_bike_lane': 5,
    # 'change_lane_boundary_dash_solid': 6
}



def polygon_pt_dist(polygon: Polygon, pt: Point, show_plot: bool = False) -> float:
    """Returns polygon to point distance

    Args:
        polygon: as a shapely Polygon
        pt: as a shapely Point
        show_plot: boolean indicating whether to visualize the objects using Matplotlib.

    Returns:
        dist: float representing distance ...
    """
    pol_ext = LinearRing(polygon.exterior.coords)
    # distance along the ring to a point nearest the other object.
    d_geodesic = pol_ext.project(pt)
    # return a point at the specified distance along the ring
    nearest_p = pol_ext.interpolate(d_geodesic)
    diff = np.array(nearest_p.coords) - np.array(pt)
    dist = np.linalg.norm(diff)

    if show_plot:
        pt_coords = np.array(pt.coords).squeeze()
        nearest_p_coords = np.array(nearest_p.coords).squeeze()

        plt.scatter(pt_coords[0], pt_coords[1], 10, color="k")
        plt.scatter(nearest_p_coords[0], nearest_p_coords[1], 10, color="r")

        plt.plot(*polygon.exterior.xy, color="b")
        plt.show()

    return dist


class SpatialMapChangeEvent(NamedTuple):
    """Object that stores a specific, spatially-localized area of a map that has undergone a real-world change.

    Args:
        log_id: unique ID of TbV vehicle log.
        change_type:
        city_coords:
        visible_line_segment: 
    """

    log_id: str
    change_type: str
    city_coords: np.ndarray  # for point object like crosswalk
    visible_line_segment: np.ndarray = None  # for line object like lane

    def check_if_in_range(self, log_id: str, query_city_coords: int, range_thresh_m: float) -> bool:
        """Determine whether ???  

        Args:
            log_id: unique ID of TbV vehicle log.
            query_city_coords:
            range_thresh_m: maximum range (in meters) to use for evaluation. Map entities found
                beyond this distance will not be considered for evaluation.

        Returns:
            Whether ...
        """
        # look up pose at this time beforehand
        if self.city_coords is not None:
            dist = np.linalg.norm(query_city_coords - self.city_coords, ord=np.inf)

        elif self.visible_line_segment is not None:
            polygon = np.vstack([self.visible_line_segment, self.visible_line_segment[0]])
            dist = polygon_pt_dist(Polygon(polygon), Point(query_city_coords))

        return dist < range_thresh_m

    def get_entity_city_pts_Nx2(self) -> np.ndarray:
        """Convert ??? entity to a 2-d set of coordinates.

        Returns:
            Array of shape (N,2) representing a 2d map entity.
        """
        if self.city_coords is not None:
            return self.city_coords.reshape(1, 2)

        # otherwise, must be a line object
        assert self.visible_line_segment is not None
        polygon = np.vstack([self.visible_line_segment, self.visible_line_segment[0]])
        return polygon.reshape(-1, 2)

    def check_if_in_range_egoview(
        self, log_id: str, query_city_coords: int, range_thresh_m: float, ego_metadata: EgoViewMapRenderer
    ) -> bool:
        """Determine whether 

        Args:
            log_id: unique ID of TbV vehicle log.
            query_city_coords
            range_thresh_m: maximum range (in meters) to use for evaluation. Map entities found
                beyond this distance will not be considered for evaluation.
            ego_metadata: contains pose, camera extrinsics, and camera intrinsics

        Returns:
            boolean indicating ...
        """
        is_nearby = self.check_if_in_range(
            log_id=log_id, query_city_coords=query_city_coords, range_thresh_m=range_thresh_m
        )

        # TODO: interp points
        entity_cityfr = self.get_entity_city_pts_Nx2()
        # lift to 3D
        entity_cityfr = ego_metadata.avm.append_height_to_2d_city_pt_cloud(pt_cloud_xy=entity_cityfr)
        entity_egofr = ego_metadata.ego_SE3_city.transform_point_cloud(entity_cityfr)
        _, _, valid_pts_bool = ego_metadata.pinhole_cam.project_ego_to_img(points_ego=entity_egofr, remove_nan=False)

        is_visible = valid_pts_bool.sum() > 0
        if not is_visible:
            logging.info(
                "None of the changed points "
                + str(np.round(entity_egofr.mean(axis=0)))
                + f"projected into frustum {ego_metadata.camera_name}"
            )

        return is_nearby and is_visible


def get_test_set_event_info_bev(data_root: Path, avm: ArgoverseStaticMap) -> Dict[str, List[SpatialMapChangeEvent]]:
    """Load GT labels for test set from disk, and convert them to SpatialMapChangeEvent objects per log.

    Args:
        data_root

    Returns:
        logid_to_mc_events_dict: dictionary from log_id to associated annotated GT map change events.
    """
    localization_data_fpath = "labeled_data/mcd_test_set_localization_in_space.json"
    localization_data = io_utils.read_json_file(localization_data_fpath)

    logid_to_mc_events_dict = defaultdict(list)

    for log_data in localization_data:
        log_id = log_data["log_id"]
        # init an empty list, in case it was a `positive` before or after log
        logid_to_mc_events_dict[log_id] = []
        for event_data in log_data["events"]:

            if event_data["supercategory"] == "crosswalk_change":
                sensor_change_type = event_data["change_type"]
                map_change_type = DUAL_CATEGORY_DICT[sensor_change_type]

                # TODO: lift 2d coordinates to 3d.

                mc_event = SpatialMapChangeEvent(
                    log_id=log_id,
                    change_type=map_change_type,
                    city_coords=np.array(event_data["City Coords"]),
                    visible_line_segment=None,
                )
                logid_to_mc_events_dict[log_id] += [mc_event]

            elif event_data["supercategory"] == "lane_geometry_change":
                # print(log_id)
                sensor_change_type = event_data["change_type"]
                map_change_type = DUAL_CATEGORY_DICT[sensor_change_type]
                map_change_type = "change_lane_marking_color"

                visible_line_segment = []
                for waypt_dict in event_data["change_endpoints"]:

                    # TODO: lift 2d coordinates to 3d.
                    waypt = np.array(waypt_dict["City Coords"])
                    visible_line_segment += [waypt]
                visible_line_segment = np.array(visible_line_segment)
                assert visible_line_segment.shape[0] >= 2  # need at least 2 waypoints

                mc_event = SpatialMapChangeEvent(
                    log_id=log_id,
                    change_type=map_change_type,
                    city_coords=None,
                    visible_line_segment=visible_line_segment,
                )
                logid_to_mc_events_dict[log_id] += [mc_event]
            else:
                raise NotImplementedError

    return logid_to_mc_events_dict


def calc_ap(gt_ranked: np.ndarray, recalls_interp: np.ndarray, ninst: int) -> Tuple[float, np.ndarray]:
    """Compute precision and recall, interpolated over n fixed recall points.
    Args:
        gt_ranked: Ground truths, ranked by confidence.
        recalls_interp: Interpolated recall values.
        ninst: Number of instances of this class.

    Returns:
        avg_precision: Average precision.
        precisions_interp: Interpolated precision values.
    """
    tp = gt_ranked

    cumulative_tp = np.cumsum(tp, dtype=np.int)
    cumulative_fp = np.cumsum(~tp, dtype=np.int)
    cumulative_fn = ninst - cumulative_tp

    precisions = cumulative_tp / (cumulative_tp + cumulative_fp + np.finfo(float).eps)
    recalls = cumulative_tp / (cumulative_tp + cumulative_fn)
    precisions = interp(precisions)
    precisions_interp = np.interp(recalls_interp, recalls, precisions, right=0)
    avg_precision = precisions_interp.mean()
    return avg_precision, precisions_interp


def plot_pr_curve_sklearn(split: str, all_gts: np.ndarray, all_pred_dists: np.ndarray):
    """ """
    all_probs = 1 / all_pred_dists

    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(all_gts, all_probs)
    pdb.set_trace()
    plt.plot(recall, precision)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(f"PR Curve on {split} set")
    plt.show()


def plot_pr_curve_numpy(split: str, all_gts: np.ndarray, all_pred_dists: np.ndarray):
    """ """
    all_probs = 1 / all_pred_dists
    pdb.set_trace()

    map_am = MeanAveragePrecisionAvgMeter()
    map_am.update(all_probs, all_gts)

    plt.title(f"AP: {ap}")
    plt.plot(recalls_interp, precisions_interp)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.show()

    return ap


def test_plot_pr_curve_numpy():
    """ """
    split = "val"
    # positive should be when it is a match
    # negative should be when it is a mismatch
    #                           TP    FP  TN   TP
    all_gts = np.array([1, 0, 0, 1])
    all_pred_dists = np.array([0.9, 0.3, 1.5, 0.5])
    plot_pr_curve_numpy(split, all_gts, all_pred_dists)


def interp(prec: np.ndarray) -> np.ndarray:
    """Interpolate the precision over all recall levels.
    Args:
            prec: Precision at all recall levels (N, ).
            method: Accumulation method.
    Returns:
            prec_interp: Interpolated precision at all recall levels (N,).
    """
    prec_interp = np.maximum.accumulate(prec[::-1])[::-1]
    return prec_interp


@click.command(help="Evaluate predictions on the val or test split of the TbV Dataset.")
@click.option(
    "-d",
    "--data-root",
    required=True,
    help="Path to local directory where the Argoverse 2 Sensor Dataset logs are stored.",
    type=click.Path(exists=True),
)
def run_evaluate_map_change_detection() -> None:
    """Click entry point for ... """
    get_test_set_event_info_bev(data_root)


if __name__ == "__main__":
    run_evaluate_map_change_detection()
