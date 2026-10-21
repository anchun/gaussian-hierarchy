#include "spatial_merge.h"
#include "dependencies/json.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

using json = nlohmann::json;

namespace
{
int integer(const json& value, const char* name, int minimum, int maximum)
{
	if (!value.is_number_integer() || value < minimum || value > maximum)
		throw std::runtime_error(std::string("invalid integer ") + name);
	return value.get<int>();
}

double number(const json& value, const char* name)
{
	if (!value.is_number() || !std::isfinite(value.get<double>()))
		throw std::runtime_error(std::string("invalid finite number ") + name);
	return value.get<double>();
}

std::string digest(const json& value, const char* name)
{
	if (!value.is_string())
		throw std::runtime_error(std::string("invalid digest ") + name);
	const std::string result = value.get<std::string>();
	if (result.size() != 64 || result.find_first_not_of("0123456789abcdef") != std::string::npos)
		throw std::runtime_error(std::string("invalid digest ") + name);
	return result;
}
}

bool loadSpatialChunkInfo(const std::string& path, SpatialChunkInfo& info, std::string& error)
{
	info = SpatialChunkInfo{};
	try
	{
		if (!std::filesystem::exists(path))
			return true;
		std::ifstream file(path);
		if (!file)
			throw std::runtime_error("cannot open metadata");
		const std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		const json value = json::parse(contents, nullptr, false);
		bool marked = false;
		for (const char* key : {"partition_mode", "partition_tree", "partition_id", "selection_id", "schema_version"})
			marked = marked || (value.is_object() && value.contains(key))
				|| contents.find(std::string("\"") + key + "\"") != std::string::npos;
		// Unmarked legacy metadata remains the responsibility of the unchanged route parser.
		// Malformed marked input (or an unmarked member of an otherwise spatial set) must fail.
		if (!marked)
			return true;
		if (value.is_discarded())
			throw std::runtime_error("cannot parse spatial JSON");
		if (!value.is_object())
			throw std::runtime_error("metadata must be an object");
		if (value.at("partition_mode") != "spatial_bsp")
			throw std::runtime_error("unsupported spatial partition mode");
		integer(value.at("schema_version"), "schema_version", 1, 1);
		info.partition_id = digest(value.at("partition_id"), "partition_id");
		info.selection_id = digest(value.at("selection_id"), "selection_id");
		info.chunk_count = integer(value.at("chunk_count"), "chunk_count", 1, std::numeric_limits<int>::max() / 2);
		info.chunk_id = integer(value.at("chunk_id"), "chunk_id", 0, info.chunk_count - 1);
		info.overlap = number(value.at("overlap_m"), "overlap_m");
		info.context_distance = number(value.at("context_distance_m"), "context_distance_m");
		info.target_chunk_size = number(value.at("target_chunk_size_m"), "target_chunk_size_m");
		if (info.overlap < 0. || info.context_distance < 0. || info.target_chunk_size <= 0.
			|| (info.context_distance > 0. && info.context_distance <= info.overlap))
			throw std::runtime_error("invalid spatial selection distances");
		info.context_frame_stride = integer(value.at("context_frame_stride"),
			"context_frame_stride", 1, std::numeric_limits<int>::max());
		const json& frames = value.at("core_frame_names");
		if (!frames.is_array() || frames.empty())
			throw std::runtime_error("core_frame_names must be a non-empty array");
		for (const json& frame : frames)
		{
			if (!frame.is_string())
				throw std::runtime_error("invalid core frame name");
			const std::string name = frame.get<std::string>();
			if (name.empty() || std::filesystem::path(name).filename().string() != name
				|| !info.core_frame_names.insert(name).second)
				throw std::runtime_error("invalid or repeated core frame name");
		}
		if (value.contains("partition_frame"))
		{
			const json& frame = value.at("partition_frame");
			if (!frame.is_object() || frame.size() != 1 || !frame.at("axis_xy").is_array()
				|| frame.at("axis_xy").size() != 2)
				throw std::runtime_error("invalid spatial partition frame");
			info.axis = Eigen::Vector2d(number(frame.at("axis_xy")[0], "axis_xy"),
				number(frame.at("axis_xy")[1], "axis_xy"));
			if (std::abs(info.axis.norm() - 1.) > 1e-9)
				throw std::runtime_error("spatial partition axis must be a unit vector");
			info.rotated = true;
		}
		const json& tree = value.at("partition_tree");
		if (!tree.is_array() || tree.size() != static_cast<std::size_t>(2 * info.chunk_count - 1))
			throw std::runtime_error("invalid spatial partition tree size");
		info.tree.resize(tree.size());
		struct Visit
		{
			int index;
			Eigen::Vector2d lower;
			Eigen::Vector2d upper;
		};
		const double infinity = std::numeric_limits<double>::infinity();
		std::vector<Visit> pending{{0, Eigen::Vector2d::Constant(-infinity), Eigen::Vector2d::Constant(infinity)}};
		std::vector<bool> seen(tree.size(), false);
		std::vector<bool> leaves(info.chunk_count, false);
		std::size_t visited = 0;
		while (!pending.empty())
		{
			const Visit visit = pending.back();
			pending.pop_back();
			if (seen[visit.index])
				throw std::runtime_error("spatial tree has a cycle or shared child");
			seen[visit.index] = true;
			++visited;
			const json& node = tree[visit.index];
			if (!node.is_object())
				throw std::runtime_error("invalid spatial tree node");
			SpatialPartitionNode& output = info.tree[visit.index];
			if (node.contains("chunk_id"))
			{
				if (node.size() != 1)
					throw std::runtime_error("invalid spatial leaf fields");
				output.chunk_id = integer(node.at("chunk_id"), "leaf chunk_id", 0, info.chunk_count - 1);
				if (leaves[output.chunk_id])
					throw std::runtime_error("repeated spatial leaf id");
				leaves[output.chunk_id] = true;
			}
			else
			{
				if (node.size() != 4)
					throw std::runtime_error("invalid spatial split fields");
				output.axis = integer(node.at("axis"), "axis", 0, 1);
				output.cut = number(node.at("cut"), "cut");
				output.left = integer(node.at("left"), "left", 0, static_cast<int>(tree.size()) - 1);
				output.right = integer(node.at("right"), "right", 0, static_cast<int>(tree.size()) - 1);
				if (!(visit.lower[output.axis] < output.cut && output.cut < visit.upper[output.axis]))
					throw std::runtime_error("spatial split creates an empty region");
				Eigen::Vector2d left_upper = visit.upper;
				Eigen::Vector2d right_lower = visit.lower;
				left_upper[output.axis] = right_lower[output.axis] = output.cut;
				pending.push_back({output.right, right_lower, visit.upper});
				pending.push_back({output.left, visit.lower, left_upper});
			}
		}
		if (visited != tree.size() || std::find(leaves.begin(), leaves.end(), false) != leaves.end())
			throw std::runtime_error("spatial tree has unreachable nodes or missing leaves");
		info.metadata = value.dump();
		info.spatial = true;
		return true;
	}
	catch (const std::exception& exception)
	{
		error = "invalid spatial metadata in " + path + ": " + exception.what();
		return false;
	}
}

bool validateSpatialTraining(
	const std::string& source_directory, const std::string& trained_directory,
	SpatialChunkInfo& info, std::string& error)
{
	SpatialChunkInfo trained;
	if (!loadSpatialChunkInfo(source_directory + "/chunk_info.json", info, error)
		|| !loadSpatialChunkInfo(trained_directory + "/chunk_info.json", trained, error))
		return false;
	if (info.spatial != trained.spatial)
	{
		error = info.spatial ? "missing spatial training snapshot in " + trained_directory
			: "spatial training snapshot cannot be used with legacy source " + source_directory;
		return false;
	}
	if (info.spatial && json::parse(info.metadata) != json::parse(trained.metadata))
	{
		error = "spatial source/training snapshot mismatch in " + trained_directory;
		return false;
	}
	return true;
}

bool validateSpatialChunkSet(
	const std::vector<std::string>& source_directories,
	const std::vector<std::string>& trained_directories,
	std::vector<SpatialChunkInfo>& infos, bool& spatial, std::string& error)
{
	spatial = false;
	infos.clear();
	if (source_directories.empty() || source_directories.size() != trained_directories.size())
	{
		error = "empty or inconsistent chunk directory set";
		return false;
	}
	infos.resize(source_directories.size());
	for (std::size_t index = 0; index < infos.size(); ++index)
	{
		if (!validateSpatialTraining(source_directories[index], trained_directories[index], infos[index], error))
			return false;
		spatial = spatial || infos[index].spatial;
	}
	if (!spatial)
		return true;
	std::unordered_set<int> ids;
	std::unordered_set<std::string> frames;
	const json reference = infos[0].spatial ? json::parse(infos[0].metadata) : json{};
	for (const SpatialChunkInfo& info : infos)
	{
		if (!info.spatial)
		{
			error = "cannot mix spatial and legacy chunks";
			return false;
		}
		if (info.chunk_count != static_cast<int>(infos.size()) || !ids.insert(info.chunk_id).second)
		{
			error = "spatial chunk set is incomplete or contains duplicate ids";
			return false;
		}
		const json value = json::parse(info.metadata);
		for (const char* key : {"partition_id", "partition_tree", "chunk_count", "overlap_m",
			"context_distance_m", "context_frame_stride", "target_chunk_size_m"})
		{
			if (!reference.contains(key) || value.at(key) != reference.at(key))
			{
				error = std::string("inconsistent spatial ") + key;
				return false;
			}
		}
		if (value.contains("partition_frame") != reference.contains("partition_frame")
			|| (value.contains("partition_frame") && value.at("partition_frame") != reference.at("partition_frame")))
		{
			error = "inconsistent spatial partition_frame";
			return false;
		}
		for (const std::string& frame : info.core_frame_names)
		{
			if (!frames.insert(frame).second)
			{
				error = "spatial chunks contain duplicate core rig frames";
				return false;
			}
		}
	}
	return true;
}

int spatialOwner(const SpatialChunkInfo& info, const Eigen::Vector2d& position)
{
	if (!info.spatial || info.tree.empty() || !position.allFinite())
		throw std::runtime_error("spatial ownership requires validated metadata and finite XY");
	// Same operation order as preprocess/spatial_chunks.py partition_xy.
	const double x = position.x(), y = position.y(), ax = info.axis.x(), ay = info.axis.y();
	const Eigen::Vector2d xy = info.rotated
		? Eigen::Vector2d(ax * x + ay * y, -ay * x + ax * y) : position;
	int index = 0;
	for (std::size_t steps = 0; steps < info.tree.size(); ++steps)
	{
		const SpatialPartitionNode& node = info.tree.at(index);
		if (node.chunk_id >= 0)
			return node.chunk_id;
		index = xy[node.axis] < node.cut ? node.left : node.right;
	}
	throw std::runtime_error("invalid spatial ownership tree");
}
