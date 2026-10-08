#include "merger_policy.h"
#include "dependencies/json.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <unordered_map>

using json = nlohmann::json;

namespace
{
constexpr float kShC0 = 0.28209479177387814f;
constexpr float kShC1 = 0.4886025119029199f;
constexpr float kRobustSigma = 6.f;
constexpr float kMinLuminanceGap = 0.05f;
constexpr float kMinAmplitudeGap = 0.02f;

float luminance(const Eigen::Vector3f& rgb)
{
	return 0.2126f * rgb.x() + 0.7152f * rgb.y() + 0.0722f * rgb.z();
}

float median(std::vector<float> values)
{
	if (values.empty())
		return 0.f;
	const std::size_t middle = values.size() / 2;
	std::nth_element(values.begin(), values.begin() + middle, values.end());
	const float upper = values[middle];
	if (values.size() % 2 != 0)
		return upper;
	std::nth_element(values.begin(), values.begin() + middle - 1, values.begin() + middle);
	return 0.5f * (values[middle - 1] + upper);
}

float medianAbsoluteDeviation(const std::vector<float>& values, float center)
{
	std::vector<float> deviations;
	deviations.reserve(values.size());
	for (const float value : values)
		deviations.push_back(std::abs(value - center));
	return median(std::move(deviations));
}

Eigen::Vector3f coefficient(const Gaussian& gaussian, int index)
{
	return gaussian.shs.segment<3>(index * 3);
}

float degreeOneDirectionalAmplitude(const Gaussian& gaussian)
{
	const float x = -kShC1 * luminance(coefficient(gaussian, 3));
	const float y = -kShC1 * luminance(coefficient(gaussian, 1));
	const float z = kShC1 * luminance(coefficient(gaussian, 2));
	return Eigen::Vector3f(x, y, z).norm();
}

float degreeOneMinimumLuminance(const Gaussian& gaussian)
{
	const float base = 0.5f + kShC0 * luminance(coefficient(gaussian, 0));
	return base - degreeOneDirectionalAmplitude(gaussian);
}

Eigen::Vector3f boxMinimum(const Box& box)
{
	return box.minn.head<3>();
}

Eigen::Vector3f boxMaximum(const Box& box)
{
	return box.maxx.head<3>();
}

Eigen::Vector3f boxCenter(const Box& box)
{
	return 0.5f * (boxMinimum(box) + boxMaximum(box));
}

float boxGap(const Box& left, const Box& right)
{
	const Eigen::Vector3f gap = (boxMinimum(left) - boxMaximum(right))
		.cwiseMax(boxMinimum(right) - boxMaximum(left)).cwiseMax(0.f);
	return gap.norm();
}

float boxOverlapCoverage(const Box& left, const Box& right)
{
	const Eigen::Vector3f intersection = boxMaximum(left).cwiseMin(boxMaximum(right))
		- boxMinimum(left).cwiseMax(boxMinimum(right));
	if ((intersection.array() <= 0.f).any())
		return 0.f;
	const float intersection_volume = intersection.prod();
	const float left_volume = (boxMaximum(left) - boxMinimum(left)).prod();
	const float right_volume = (boxMaximum(right) - boxMinimum(right)).prod();
	const float smaller_volume = std::min(left_volume, right_volume);
	return smaller_volume <= 0.f ? 0.f : intersection_volume / smaller_volume;
}

Box unionBox(const Box& left, const Box& right)
{
	return Box(boxMinimum(left).cwiseMin(boxMinimum(right)),
		boxMaximum(left).cwiseMax(boxMaximum(right)));
}

struct GridKey
{
	int x;
	int y;
	int z;

	bool operator==(const GridKey& other) const
	{
		return x == other.x && y == other.y && z == other.z;
	}
};

struct GridKeyHash
{
	std::size_t operator()(const GridKey& key) const
	{
		std::size_t result = std::hash<int>{}(key.x);
		result ^= std::hash<int>{}(key.y) + 0x9e3779b9 + (result << 6) + (result >> 2);
		result ^= std::hash<int>{}(key.z) + 0x9e3779b9 + (result << 6) + (result >> 2);
		return result;
	}
};

Eigen::Vector3i minimumCell(const Box& box, float padding, float cell_size)
{
	return ((boxMinimum(box).array() - padding) / cell_size).floor().cast<int>();
}

Eigen::Vector3i maximumCell(const Box& box, float padding, float cell_size)
{
	return ((boxMaximum(box).array() + padding) / cell_size).floor().cast<int>();
}

std::size_t cellCount(const Eigen::Vector3i& minimum, const Eigen::Vector3i& maximum)
{
	const Eigen::Array3i size = (maximum - minimum + Eigen::Vector3i::Ones()).array();
	return static_cast<std::size_t>(size.x()) * static_cast<std::size_t>(size.y())
		* static_cast<std::size_t>(size.z());
}

float routePosition(const Eigen::Vector3f& point, const ChunkRouteInfo& route)
{
	return (point.head<2>() - route.origin).dot(route.axis);
}

float supportScore(
	const Box& bounds,
	const ChunkRouteInfo& route,
	const std::vector<CameraSupport>& cameras)
{
	if (cameras.empty())
		return 0.f;
	const Eigen::Vector3f center = boxCenter(bounds);
	const float radius = 0.5f * (boxMaximum(bounds) - boxMinimum(bounds)).norm();
	const float center_route = routePosition(center, route);
	const float route_radius = std::max(route.context_distance, 2.f * route.overlap);
	float core_sum = 0.f;
	float context_sum = 0.f;
	std::size_t core_count = 0;
	std::size_t context_count = 0;
	for (const CameraSupport& camera : cameras)
	{
		if (std::abs(camera.route_position - center_route) > route_radius)
			continue;
		if (camera.core)
			++core_count;
		else
			++context_count;
		const Eigen::Vector3f camera_point = camera.rotation.transpose() * (center - camera.position);
		if (camera_point.z() + radius <= 0.f)
			continue;
		const float depth = std::max(camera_point.z(), 1e-3f);
		const float horizontal_limit = depth * camera.width / (2.f * camera.fx) + radius;
		const float vertical_limit = depth * camera.height / (2.f * camera.fy) + radius;
		if (std::abs(camera_point.x()) > horizontal_limit
			|| std::abs(camera_point.y()) > vertical_limit)
			continue;
		const float projected_radius = camera.fx * radius / std::max(depth - radius, 1.f);
		const float footprint = std::clamp(projected_radius / 32.f, 0.1f, 1.f);
		const float distance_weight = 1.f / (1.f + (center - camera.position).norm() / 200.f);
		if (camera.core)
			core_sum += footprint * distance_weight;
		else
			context_sum += footprint * distance_weight;
	}
	const float core_score = core_count == 0 ? 0.f : core_sum / static_cast<float>(core_count);
	const float context_score = context_count == 0 ? 0.f : context_sum / static_cast<float>(context_count);
	return core_score + 0.25f * context_score;
}

bool readJson(const std::string& path, json& value, std::string& error)
{
	std::ifstream file(path);
	if (!file.good())
	{
		error = "cannot open " + path;
		return false;
	}
	try
	{
		file >> value;
	}
	catch (const std::exception& exception)
	{
		error = "cannot parse " + path + ": " + exception.what();
		return false;
	}
	return true;
}
}

bool loadPriorInfo(
	const std::string& path,
	std::size_t scaffold_points,
	PriorInfo& info,
	std::string& error)
{
	json value;
	if (!readJson(path, value, error))
		return false;
	try
	{
		info.road_points = value.at("road_points").get<std::size_t>();
		info.skybox_points = value.at("skybox_points").get<std::size_t>();
		info.fixed_prior_points = value.at("fixed_prior_points").get<std::size_t>();
	}
	catch (const std::exception& exception)
	{
		error = "invalid prior metadata in " + path + ": " + exception.what();
		return false;
	}
	if (info.fixed_prior_points != info.road_points + info.skybox_points)
	{
		error = "fixed_prior_points does not equal road_points + skybox_points";
		return false;
	}
	if (info.fixed_prior_points > scaffold_points)
	{
		error = "fixed prior range exceeds loaded scaffold prefix";
		return false;
	}
	return true;
}

bool loadChunkRouteInfo(
	const std::string& path,
	ChunkRouteInfo& info,
	std::string& error)
{
	json value;
	if (!readJson(path, value, error))
		return false;
	try
	{
		if (value.at("route_mode").get<std::string>() != "corridor" || value.at("route_closed").get<bool>())
		{
			error = "route is not an open corridor";
			return false;
		}
		info.start = value.at("route_start_m").get<float>();
		info.end = value.at("route_end_m").get<float>();
		info.overlap = value.at("overlap_m").get<float>();
		info.context_distance = value.at("context_distance_m").get<float>();
		info.context_frame_stride = value.at("context_frame_stride").get<int>();
		const std::string requested_mode = value.value("route_mode_requested", std::string("auto"));
		info.ambiguous = value.contains("route_ambiguous")
			? value.at("route_ambiguous").get<bool>()
			: requested_mode == "auto";
		info.explicitly_requested = requested_mode == "corridor";
		const std::vector<float> axis = value.at("route_axis_xy").get<std::vector<float>>();
		const std::vector<float> origin = value.at("route_origin_xy").get<std::vector<float>>();
		if (axis.size() != 2 || origin.size() != 2)
			throw std::runtime_error("route axis and origin must contain two values");
		info.axis = Eigen::Vector2f(axis[0], axis[1]);
		info.origin = Eigen::Vector2f(origin[0], origin[1]);
	}
	catch (const std::exception& exception)
	{
		error = "invalid route metadata in " + path + ": " + exception.what();
		return false;
	}
	if (!(info.start < info.end) || info.overlap < 0.f || info.context_distance < 0.f
		|| info.context_frame_stride < 1 || info.axis.norm() <= 1e-6f)
	{
		error = "route interval, overlap, or axis is invalid";
		return false;
	}
	info.axis.normalize();
	return true;
}

bool loadCameraSupport(
	const std::string& path,
	const ChunkRouteInfo& route,
	std::vector<CameraSupport>& cameras,
	std::string& error)
{
	json values;
	if (!readJson(path, values, error))
		return false;
	if (!values.is_array())
	{
		error = "camera metadata is not an array in " + path;
		return false;
	}
	try
	{
		for (const json& value : values)
		{
			const std::vector<float> position = value.at("position").get<std::vector<float>>();
			const std::vector<std::vector<float>> rotation = value.at("rotation").get<std::vector<std::vector<float>>>();
			if (position.size() != 3 || rotation.size() != 3
				|| rotation[0].size() != 3 || rotation[1].size() != 3 || rotation[2].size() != 3)
				throw std::runtime_error("camera pose has invalid dimensions");
			CameraSupport camera;
			camera.position = Eigen::Vector3f(position[0], position[1], position[2]);
			for (int row = 0; row < 3; ++row)
				for (int column = 0; column < 3; ++column)
					camera.rotation(row, column) = rotation[row][column];
			camera.fx = value.at("fx").get<float>();
			camera.fy = value.at("fy").get<float>();
			camera.width = value.at("width").get<float>();
			camera.height = value.at("height").get<float>();
			camera.route_position = routePosition(camera.position, route);
			camera.core = camera.route_position >= route.start && camera.route_position <= route.end;
			cameras.push_back(camera);
		}
	}
	catch (const std::exception& exception)
	{
		error = "invalid camera metadata in " + path + ": " + exception.what();
		return false;
	}
	if (cameras.empty())
	{
		error = "camera metadata contains no support cameras in " + path;
		return false;
	}
	std::sort(cameras.begin(), cameras.end(), [](const CameraSupport& left, const CameraSupport& right) {
		return left.route_position < right.route_position;
	});
	return true;
}

RouteOwnershipPlan buildRouteOwnership(
	const std::vector<OwnershipUnit>& left_units,
	const std::vector<OwnershipUnit>& right_units,
	const ChunkRouteInfo& left_route,
	const ChunkRouteInfo& right_route,
	const RouteTransition& transition,
	const std::vector<CameraSupport>& left_cameras,
	const std::vector<CameraSupport>& right_cameras)
{
	RouteOwnershipPlan plan;
	plan.excluded_nodes.resize(2);
	plan.stats.left_units = left_units.size();
	plan.stats.right_units = right_units.size();
	if (left_units.empty() || right_units.empty())
	{
		plan.stats.unmatched_units = left_units.size() + right_units.size();
		return plan;
	}

	const float match_distance = std::max(0.25f, (transition.end - transition.start) / 200.f);
	const float cell_size = std::max(2.f * match_distance,
		(transition.end - transition.start) / 10.f);
	constexpr std::size_t kMaximumCellsPerUnit = 512;
	std::vector<int> best_right(left_units.size(), -1);
	std::vector<float> best_right_distance(left_units.size(), std::numeric_limits<float>::max());
	std::vector<int> best_left(right_units.size(), -1);
	std::vector<float> best_left_distance(right_units.size(), std::numeric_limits<float>::max());
	std::unordered_map<GridKey, std::vector<int>, GridKeyHash> right_grid;
	std::vector<int> large_right_units;
	for (std::size_t right_index = 0; right_index < right_units.size(); ++right_index)
	{
		const Eigen::Vector3i minimum = minimumCell(right_units[right_index].bounds, match_distance, cell_size);
		const Eigen::Vector3i maximum = maximumCell(right_units[right_index].bounds, match_distance, cell_size);
		if (cellCount(minimum, maximum) > kMaximumCellsPerUnit)
		{
			large_right_units.push_back(static_cast<int>(right_index));
			continue;
		}
		for (int x = minimum.x(); x <= maximum.x(); ++x)
			for (int y = minimum.y(); y <= maximum.y(); ++y)
				for (int z = minimum.z(); z <= maximum.z(); ++z)
					right_grid[{x, y, z}].push_back(static_cast<int>(right_index));
	}
	std::vector<int> seen(right_units.size(), -1);
	for (std::size_t left_index = 0; left_index < left_units.size(); ++left_index)
	{
		std::vector<int> candidates = large_right_units;
		const Eigen::Vector3i minimum = minimumCell(left_units[left_index].bounds, 0.f, cell_size);
		const Eigen::Vector3i maximum = maximumCell(left_units[left_index].bounds, 0.f, cell_size);
		if (cellCount(minimum, maximum) > kMaximumCellsPerUnit)
		{
			candidates.resize(right_units.size());
			for (std::size_t index = 0; index < right_units.size(); ++index)
				candidates[index] = static_cast<int>(index);
		}
		else
		{
			for (int x = minimum.x(); x <= maximum.x(); ++x)
				for (int y = minimum.y(); y <= maximum.y(); ++y)
					for (int z = minimum.z(); z <= maximum.z(); ++z)
					{
						const auto found = right_grid.find({x, y, z});
						if (found != right_grid.end())
							candidates.insert(candidates.end(), found->second.begin(), found->second.end());
					}
		}
		for (const int right_index : candidates)
		{
			if (seen[right_index] == static_cast<int>(left_index))
				continue;
			seen[right_index] = static_cast<int>(left_index);
			const float gap = boxGap(left_units[left_index].bounds, right_units[right_index].bounds);
			if (gap > match_distance
				|| boxOverlapCoverage(left_units[left_index].bounds, right_units[right_index].bounds) < 0.9f)
				continue;
			const float distance = gap + 0.01f * (boxCenter(left_units[left_index].bounds)
				- boxCenter(right_units[right_index].bounds)).norm();
			if (distance < best_right_distance[left_index])
			{
				best_right_distance[left_index] = distance;
				best_right[left_index] = static_cast<int>(right_index);
			}
			if (distance < best_left_distance[right_index])
			{
				best_left_distance[right_index] = distance;
				best_left[right_index] = static_cast<int>(left_index);
			}
		}
	}

	for (std::size_t left_index = 0; left_index < left_units.size(); ++left_index)
	{
		const int right_index = best_right[left_index];
		if (right_index < 0 || best_left[right_index] != static_cast<int>(left_index))
			continue;
		++plan.stats.matched_pairs;
		const Box bounds = unionBox(
			left_units[left_index].bounds, right_units[right_index].bounds);
		const float center_route = routePosition(boxCenter(bounds), left_route);
		bool left_owns;
		if (center_route < transition.start)
		{
			left_owns = true;
			++plan.stats.matched_before_transition;
		}
		else if (center_route > transition.end)
		{
			left_owns = false;
			++plan.stats.matched_after_transition;
		}
		else
		{
			++plan.stats.matched_in_transition;
			left_owns = center_route <= transition.seam;
			if (std::abs(center_route - transition.seam) <= match_distance)
			{
				++plan.stats.camera_arbitrated_pairs;
				const float left_score = supportScore(bounds, left_route, left_cameras);
				const float right_score = supportScore(bounds, right_route, right_cameras);
				if (std::abs(left_score - right_score) > 1e-6f)
					left_owns = left_score > right_score;
			}
		}
		if (left_owns)
		{
			plan.excluded_nodes[1].insert(right_units[right_index].node_id);
			++plan.stats.left_owned_pairs;
		}
		else
		{
			plan.excluded_nodes[0].insert(left_units[left_index].node_id);
			++plan.stats.right_owned_pairs;
		}
	}
	plan.stats.unmatched_units = left_units.size() + right_units.size()
		- 2 * plan.stats.matched_pairs;
	return plan;
}

bool deriveRouteTransition(
	const ChunkRouteInfo& left,
	const ChunkRouteInfo& right,
	RouteTransition& transition,
	std::string& error)
{
	if (left.axis.dot(right.axis) < 0.999f || (left.origin - right.origin).norm() > 1e-3f)
	{
		error = "adjacent chunks do not share route geometry";
		return false;
	}
	if (left.start > right.start)
	{
		error = "chunks are not in route order";
		return false;
	}
	transition.seam = 0.5f * (left.end + right.start);
	const float half_width = std::min(left.overlap, right.overlap);
	transition.start = transition.seam - half_width;
	transition.end = transition.seam + half_width;
	return true;
}

SkyRepairStats repairAnomalousSky(
	std::vector<Gaussian>& scaffold,
	const PriorInfo& info)
{
	SkyRepairStats stats;
	if (info.skybox_points == 0 || info.fixed_prior_points > scaffold.size())
		return stats;

	const std::size_t begin = info.road_points;
	const std::size_t end = info.fixed_prior_points;
	stats.sky_points = end - begin;
	std::vector<float> minimum_luminances;
	std::vector<float> amplitudes;
	minimum_luminances.reserve(stats.sky_points);
	amplitudes.reserve(stats.sky_points);
	for (std::size_t index = begin; index < end; ++index)
	{
		minimum_luminances.push_back(degreeOneMinimumLuminance(scaffold[index]));
		amplitudes.push_back(degreeOneDirectionalAmplitude(scaffold[index]));
	}

	stats.median_min_luminance = median(minimum_luminances);
	stats.median_directional_amplitude = median(amplitudes);
	const float luminance_mad = medianAbsoluteDeviation(minimum_luminances, stats.median_min_luminance);
	const float amplitude_mad = medianAbsoluteDeviation(amplitudes, stats.median_directional_amplitude);
	stats.min_luminance_threshold = stats.median_min_luminance
		- std::max(kRobustSigma * 1.4826f * luminance_mad, kMinLuminanceGap);
	stats.directional_amplitude_threshold = stats.median_directional_amplitude
		+ std::max(kRobustSigma * 1.4826f * amplitude_mad, kMinAmplitudeGap);

	std::vector<std::size_t> candidates;
	std::vector<float> normal_dc[3];
	for (std::size_t offset = 0; offset < stats.sky_points; ++offset)
	{
		const bool anomalous = minimum_luminances[offset] < stats.min_luminance_threshold
			&& amplitudes[offset] > stats.directional_amplitude_threshold;
		if (anomalous)
		{
			candidates.push_back(begin + offset);
		}
		else
		{
			for (int channel = 0; channel < 3; ++channel)
				normal_dc[channel].push_back(scaffold[begin + offset].shs[channel]);
		}
	}
	if (candidates.empty() || normal_dc[0].empty())
		return stats;

	const Eigen::Vector3f replacement_dc(
		median(normal_dc[0]), median(normal_dc[1]), median(normal_dc[2]));
	for (const std::size_t index : candidates)
	{
		scaffold[index].shs.segment<3>(0) = replacement_dc;
		for (int coefficient_index = 1; coefficient_index < 16; ++coefficient_index)
			scaffold[index].shs.segment<3>(coefficient_index * 3).setZero();
	}
	stats.repaired_points = candidates.size();
	return stats;
}
