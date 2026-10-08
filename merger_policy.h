#pragma once

#include "common.h"
#include <Eigen/Dense>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

struct PriorInfo
{
	std::size_t road_points = 0;
	std::size_t skybox_points = 0;
	std::size_t fixed_prior_points = 0;
};

struct ChunkRouteInfo
{
	float start = 0.f;
	float end = 0.f;
	float overlap = 0.f;
	float context_distance = 0.f;
	int context_frame_stride = 1;
	bool ambiguous = false;
	bool explicitly_requested = false;
	Eigen::Vector2f axis = Eigen::Vector2f::Zero();
	Eigen::Vector2f origin = Eigen::Vector2f::Zero();
};

struct CameraSupport
{
	Eigen::Vector3f position = Eigen::Vector3f::Zero();
	Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
	float fx = 0.f;
	float fy = 0.f;
	float width = 0.f;
	float height = 0.f;
	float route_position = 0.f;
	bool core = false;
};

struct OwnershipUnit
{
	int node_id = -1;
	Box bounds;
};

struct RouteOwnershipStats
{
	std::size_t left_units = 0;
	std::size_t right_units = 0;
	std::size_t matched_pairs = 0;
	std::size_t matched_before_transition = 0;
	std::size_t matched_in_transition = 0;
	std::size_t matched_after_transition = 0;
	std::size_t camera_arbitrated_pairs = 0;
	std::size_t left_owned_pairs = 0;
	std::size_t right_owned_pairs = 0;
	std::size_t unmatched_units = 0;
};

struct RouteOwnershipPlan
{
	std::vector<std::unordered_set<int>> excluded_nodes;
	RouteOwnershipStats stats;
};

struct RouteTransition
{
	float seam = 0.f;
	float start = 0.f;
	float end = 0.f;
};

struct SkyRepairStats
{
	std::size_t sky_points = 0;
	std::size_t repaired_points = 0;
	float median_min_luminance = 0.f;
	float min_luminance_threshold = 0.f;
	float median_directional_amplitude = 0.f;
	float directional_amplitude_threshold = 0.f;
};

bool loadPriorInfo(
	const std::string& path,
	std::size_t scaffold_points,
	PriorInfo& info,
	std::string& error);

bool loadChunkRouteInfo(
	const std::string& path,
	ChunkRouteInfo& info,
	std::string& error);

bool deriveRouteTransition(
	const ChunkRouteInfo& left,
	const ChunkRouteInfo& right,
	RouteTransition& transition,
	std::string& error);

bool loadCameraSupport(
	const std::string& path,
	const ChunkRouteInfo& route,
	std::vector<CameraSupport>& cameras,
	std::string& error);

RouteOwnershipPlan buildRouteOwnership(
	const std::vector<OwnershipUnit>& left_units,
	const std::vector<OwnershipUnit>& right_units,
	const ChunkRouteInfo& left_route,
	const ChunkRouteInfo& right_route,
	const RouteTransition& transition,
	const std::vector<CameraSupport>& left_cameras,
	const std::vector<CameraSupport>& right_cameras);

SkyRepairStats repairAnomalousSky(
	std::vector<Gaussian>& scaffold,
	const PriorInfo& info);
