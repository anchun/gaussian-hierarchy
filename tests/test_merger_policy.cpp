#include "merger_policy.h"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace
{
constexpr float kShC0 = 0.28209479177387814f;

Gaussian makeSky(float directional)
{
	Gaussian gaussian;
	gaussian.position.setZero();
	gaussian.rotation = Eigen::Vector4f(1.f, 0.f, 0.f, 0.f);
	gaussian.scale = Eigen::Vector3f(2.f, 3.f, 4.f);
	gaussian.opacity = 0.7f;
	gaussian.shs.setZero();
	gaussian.shs.segment<3>(0) = (Eigen::Vector3f(0.7f, 0.8f, 0.95f).array() - 0.5f) / kShC0;
	gaussian.shs.segment<3>(3) = Eigen::Vector3f::Constant(directional);
	return gaussian;
}

bool close(float left, float right)
{
	return std::abs(left - right) < 1e-5f;
}

void require(bool condition, const char* message)
{
	if (!condition)
		throw std::runtime_error(message);
}
}

int main()
{
	std::vector<Gaussian> scaffold;
	scaffold.push_back(makeSky(0.f));
	for (int index = 0; index < 7; ++index)
		scaffold.push_back(makeSky(0.005f * static_cast<float>(index % 2)));
	scaffold.push_back(makeSky(4.f));

	const SHs road_sh = scaffold[0].shs;
	const Eigen::Vector3f anomaly_scale = scaffold.back().scale;
	const float anomaly_opacity = scaffold.back().opacity;
	const SkyRepairStats stats = repairAnomalousSky(scaffold, PriorInfo{1, 8, 9});
	require(stats.sky_points == 8, "sky range is incorrect");
	require(stats.repaired_points == 1, "sky anomaly count is incorrect");
	require(scaffold[0].shs == road_sh, "road SH was modified");
	require(scaffold[1].shs.segment<3>(3).norm() == 0.f, "zero normal sky SH was modified");
	require(scaffold[6].shs.segment<3>(3).norm() > 0.f, "normal directional sky SH was removed");
	require(scaffold[8].shs.segment<45>(3).norm() == 0.f, "anomalous directional SH remains");
	require(scaffold[8].scale == anomaly_scale, "sky repair changed scale");
	require(scaffold[8].opacity == anomaly_opacity, "sky repair changed opacity");

	const std::string route0_path = "/tmp/test_merger_policy_route0.json";
	const std::string route1_path = "/tmp/test_merger_policy_route1.json";
	{
		std::ofstream file(route0_path);
		file << R"({"route_mode":"corridor","route_mode_requested":"corridor","route_closed":false,"route_ambiguous":false,"route_start_m":0,"route_end_m":582,"overlap_m":50,"context_distance_m":200,"context_frame_stride":5,"route_axis_xy":[1,0],"route_origin_xy":[0,0]})";
	}
	{
		std::ofstream file(route1_path);
		file << R"({"route_mode":"corridor","route_mode_requested":"corridor","route_closed":false,"route_ambiguous":false,"route_start_m":582,"route_end_m":1164,"overlap_m":50,"context_distance_m":200,"context_frame_stride":5,"route_axis_xy":[1,0],"route_origin_xy":[0,0]})";
	}
	ChunkRouteInfo route0;
	ChunkRouteInfo route1;
	std::string error;
	require(loadChunkRouteInfo(route0_path, route0, error), "left route metadata did not load");
	require(loadChunkRouteInfo(route1_path, route1, error), "right route metadata did not load");
	require(!route0.ambiguous && !route1.ambiguous, "explicit corridor route was marked ambiguous");
	require(route0.explicitly_requested && route1.explicitly_requested,
		"explicit corridor request was not preserved");
	const std::string legacy_auto_path = "/tmp/test_merger_policy_legacy_auto.json";
	{
		std::ofstream file(legacy_auto_path);
		file << R"({"route_mode":"corridor","route_mode_requested":"auto","route_closed":false,"route_start_m":0,"route_end_m":582,"overlap_m":50,"context_distance_m":200,"context_frame_stride":5,"route_axis_xy":[1,0],"route_origin_xy":[0,0]})";
	}
	ChunkRouteInfo legacy_auto;
	require(loadChunkRouteInfo(legacy_auto_path, legacy_auto, error), "legacy auto route did not load");
	require(legacy_auto.ambiguous, "legacy auto route without ambiguity flag was not conservative");
	require(!legacy_auto.explicitly_requested, "legacy auto route became an opt-in route");
	RouteTransition transition;
	require(deriveRouteTransition(route0, route1, transition, error), "route transition was not derived");
	require(close(transition.seam, 582.f), "route seam is incorrect");
	require(close(transition.start, 532.f), "route transition start is incorrect");
	require(close(transition.end, 632.f), "route transition end is incorrect");

	const std::string prior_path = "/tmp/test_merger_policy_prior.json";
	{
		std::ofstream file(prior_path);
		file << R"({"road_points":10,"skybox_points":2,"fixed_prior_points":12})";
	}
	PriorInfo prior;
	require(loadPriorInfo(prior_path, 12, prior, error), "valid prior metadata did not load");
	require(prior.road_points == 10, "road prior count is incorrect");
	require(prior.skybox_points == 2, "sky prior count is incorrect");
	require(prior.fixed_prior_points == 12, "fixed prior count is incorrect");
	require(!loadPriorInfo(prior_path, 11, prior, error), "out-of-range prior metadata was accepted");
	const std::string cameras_path = "/tmp/test_merger_policy_cameras.json";
	{
		std::ofstream file(cameras_path);
		file << R"([{"img_name":"front/0010.jpg","position":[10,0,0],"rotation":[[1,0,0],[0,1,0],[0,0,1]],"fx":1000,"fy":1000,"width":1000,"height":1000},{"img_name":"front/0011.jpg","position":[700,0,0],"rotation":[[1,0,0],[0,1,0],[0,0,1]],"fx":1000,"fy":1000,"width":1000,"height":1000}])";
	}
	std::vector<CameraSupport> loaded_cameras;
	require(loadCameraSupport(cameras_path, route0, loaded_cameras, error), "camera support did not load");
	require(loaded_cameras.size() == 2, "preselected camera support was filtered again");
	require(loaded_cameras[0].core, "core camera classification is incorrect");
	require(!loaded_cameras[1].core, "context camera classification is incorrect");

	ChunkRouteInfo owner_left = route0;
	ChunkRouteInfo owner_right = route1;
	owner_left.context_distance = 250.f;
	owner_right.context_distance = 250.f;
	CameraSupport left_camera;
	left_camera.position = Eigen::Vector3f(0.f, 0.f, 0.f);
	left_camera.rotation << 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f;
	left_camera.fx = left_camera.fy = 1000.f;
	left_camera.width = left_camera.height = 1000.f;
	left_camera.route_position = 0.f;
	left_camera.core = true;
	CameraSupport right_camera = left_camera;
	right_camera.position = Eigen::Vector3f(200.f, 0.f, 0.f);
	right_camera.rotation << 0.f, 0.f, -1.f, -1.f, 0.f, 0.f, 0.f, 1.f, 0.f;
	right_camera.route_position = 200.f;
	OwnershipUnit left_matched{10, Box(Eigen::Vector3f(49.f, -1.f, -1.f), Eigen::Vector3f(51.f, 1.f, 1.f))};
	OwnershipUnit right_matched{20, Box(Eigen::Vector3f(49.1f, -1.f, -1.f), Eigen::Vector3f(51.1f, 1.f, 1.f))};
	OwnershipUnit right_unique{21, Box(Eigen::Vector3f(149.f, -1.f, -1.f), Eigen::Vector3f(151.f, 1.f, 1.f))};
	RouteOwnershipPlan ownership = buildRouteOwnership(
		{left_matched}, {right_matched, right_unique}, owner_left, owner_right,
		transition, {left_camera}, {right_camera});
	require(ownership.stats.matched_pairs == 1, "structure pair was not matched");
	require(ownership.stats.unmatched_units == 1, "unmatched unit count is incorrect");
	require(ownership.excluded_nodes[0].empty(), "camera-supported left owner was removed");
	require(ownership.excluded_nodes[1].count(20) == 1, "non-owner matched unit was retained");
	require(ownership.excluded_nodes[1].count(21) == 0, "unmatched unit was removed");

	CameraSupport seam_left = left_camera;
	CameraSupport seam_right = right_camera;
	seam_right.position = Eigen::Vector3f(600.f, 0.f, 0.f);
	seam_right.route_position = 600.f;
	seam_right.core = true;
	OwnershipUnit seam_left_unit{30,
		Box(Eigen::Vector3f(580.9f, -1.f, -1.f), Eigen::Vector3f(582.9f, 1.f, 1.f))};
	OwnershipUnit seam_right_unit{40,
		Box(Eigen::Vector3f(581.f, -1.f, -1.f), Eigen::Vector3f(583.f, 1.f, 1.f))};
	RouteOwnershipPlan seam_ownership = buildRouteOwnership(
		{seam_left_unit}, {seam_right_unit}, owner_left, owner_right,
		transition, {seam_left}, {seam_right});
	require(seam_ownership.stats.camera_arbitrated_pairs == 1,
		"seam structure did not use camera arbitration");
	require(seam_ownership.excluded_nodes[0].count(30) == 1,
		"better-supported right seam owner was not selected");

	OwnershipUnit competing_left{31,
		Box(Eigen::Vector3f(581.2f, -1.f, -1.f), Eigen::Vector3f(583.2f, 1.f, 1.f))};
	RouteOwnershipPlan one_to_one = buildRouteOwnership(
		{seam_left_unit, competing_left}, {seam_right_unit}, owner_left, owner_right,
		transition, {seam_left}, {seam_right});
	require(one_to_one.stats.matched_pairs == 1, "one right structure matched multiple left structures");
	require(one_to_one.stats.unmatched_units == 1, "unmatched competing structure was not preserved");
	require(one_to_one.excluded_nodes[0].size() + one_to_one.excluded_nodes[1].size() == 1,
		"one-to-one ownership removed more than one structure");
	return 0;
}
