#include "loader.h"
#include "writer.h"
#include "FlatGenerator.h"
#include "PointbasedKdTreeGenerator.h"
#include "AvgMerger.h"
#include "ClusterMerger.h"
#include "common.h"
#include "dependencies/json.hpp"
#include "hierarchy_explicit_loader.h"
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include "appearance_filter.h"
#include "rotation_aligner.h"
#include "hierarchy_writer.h"

#define LOD_LEVELS 6
using json = nlohmann::json;

struct TrajectoryPoint
{
	float x;
	float y;
	float z;
	size_t source_index;
};

class TrajectoryHeightLookup
{
private:
	struct KdNode
	{
		size_t point_index;
		int left = -1;
		int right = -1;
		int axis;
	};

	std::vector<TrajectoryPoint> points;
	std::vector<KdNode> nodes;

	static std::string trim(const std::string& value)
	{
		const size_t first = value.find_first_not_of(" \t\r\n");
		if (first == std::string::npos)
			return "";
		const size_t last = value.find_last_not_of(" \t\r\n");
		return value.substr(first, last - first + 1);
	}

	static std::vector<std::string> parseCsvRow(const std::string& line)
	{
		std::vector<std::string> fields;
		std::string field;
		bool quoted = false;
		for (size_t i = 0; i < line.size(); i++)
		{
			const char value = line[i];
			if (quoted)
			{
				if (value == '"' && i + 1 < line.size() && line[i + 1] == '"')
				{
					field.push_back('"');
					i++;
				}
				else if (value == '"')
					quoted = false;
				else
					field.push_back(value);
			}
			else if (value == ',')
			{
				fields.push_back(trim(field));
				field.clear();
			}
			else if (value == '"' && field.empty())
				quoted = true;
			else
				field.push_back(value);
		}
		if (quoted)
			throw std::runtime_error("Unterminated quoted CSV field");
		fields.push_back(trim(field));
		return fields;
	}

	static size_t findColumn(const std::vector<std::string>& header, const std::string& name)
	{
		auto column = std::find(header.begin(), header.end(), name);
		if (column == header.end())
			throw std::runtime_error("Missing trajectories.csv column: " + name);
		return static_cast<size_t>(std::distance(header.begin(), column));
	}

	static float parseFloat(const std::string& value, const std::string& column, size_t line_number)
	{
		size_t consumed = 0;
		float parsed = 0;
		try
		{
			parsed = std::stof(value, &consumed);
		}
		catch (const std::exception&)
		{
			throw std::runtime_error(
				"Invalid " + column + " at trajectories.csv line " + std::to_string(line_number));
		}
		if (consumed != value.size() || !std::isfinite(parsed))
			throw std::runtime_error(
				"Invalid " + column + " at trajectories.csv line " + std::to_string(line_number));
		return parsed;
	}

	void load(const std::filesystem::path& path)
	{
		std::ifstream input(path);
		if (!input.good())
			throw std::runtime_error("Could not open trajectories.csv: " + path.string());

		std::string line;
		if (!std::getline(input, line))
			throw std::runtime_error("Empty trajectories.csv: " + path.string());
		const auto header = parseCsvRow(line);
		const size_t x_column = findColumn(header, "PositionX");
		const size_t y_column = findColumn(header, "PositionY");
		const size_t z_column = findColumn(header, "PositionZ");
		const size_t ego_column = findColumn(header, "Ego");
		const size_t required_columns = std::max({ x_column, y_column, z_column, ego_column }) + 1;

		size_t line_number = 1;
		while (std::getline(input, line))
		{
			line_number++;
			if (trim(line).empty())
				continue;
			const auto fields = parseCsvRow(line);
			if (fields.size() < required_columns)
				throw std::runtime_error(
					"Too few columns at trajectories.csv line " + std::to_string(line_number));
			if (fields[ego_column] != "Y")
				continue;
			points.push_back({
				parseFloat(fields[x_column], "PositionX", line_number),
				parseFloat(fields[y_column], "PositionY", line_number),
				parseFloat(fields[z_column], "PositionZ", line_number),
				points.size()
			});
		}
		if (points.empty())
			throw std::runtime_error("No rows with Ego=Y in trajectories.csv: " + path.string());
	}

	int build(std::vector<size_t>& indices, size_t begin, size_t end, int depth)
	{
		if (begin == end)
			return -1;
		const int axis = depth % 2;
		const size_t middle = begin + (end - begin) / 2;
		std::nth_element(
			indices.begin() + begin,
			indices.begin() + middle,
			indices.begin() + end,
			[this, axis](size_t lhs, size_t rhs) {
				const float lhs_value = axis == 0 ? points[lhs].x : points[lhs].y;
				const float rhs_value = axis == 0 ? points[rhs].x : points[rhs].y;
				return lhs_value == rhs_value ? lhs < rhs : lhs_value < rhs_value;
			});

		const int node_index = static_cast<int>(nodes.size());
		nodes.push_back({ indices[middle], -1, -1, axis });
		nodes[node_index].left = build(indices, begin, middle, depth + 1);
		nodes[node_index].right = build(indices, middle + 1, end, depth + 1);
		return node_index;
	}

	void nearest(
		int node_index,
		float x,
		float y,
		float& best_distance_squared,
		size_t& best_point_index) const
	{
		if (node_index < 0)
			return;
		const KdNode& node = nodes[node_index];
		const TrajectoryPoint& point = points[node.point_index];
		const float dx = x - point.x;
		const float dy = y - point.y;
		const float distance_squared = dx * dx + dy * dy;
		if (distance_squared < best_distance_squared ||
			(distance_squared == best_distance_squared && node.point_index < best_point_index))
		{
			best_distance_squared = distance_squared;
			best_point_index = node.point_index;
		}

		const float delta = node.axis == 0 ? dx : dy;
		const int near_node = delta <= 0 ? node.left : node.right;
		const int far_node = delta <= 0 ? node.right : node.left;
		nearest(near_node, x, y, best_distance_squared, best_point_index);
		if (delta * delta <= best_distance_squared)
			nearest(far_node, x, y, best_distance_squared, best_point_index);
	}

public:
	explicit TrajectoryHeightLookup(const std::filesystem::path& path)
	{
		load(path);
		std::vector<size_t> indices(points.size());
		std::iota(indices.begin(), indices.end(), 0);
		nodes.reserve(points.size());
		build(indices, 0, indices.size(), 0);
	}

	bool isBelowTrajectory(const Gaussian& gaussian, float height_offset) const
	{
		if (!gaussian.position.allFinite())
			return false;
		float best_distance_squared = std::numeric_limits<float>::infinity();
		size_t best_point_index = std::numeric_limits<size_t>::max();
		nearest(0, gaussian.position.x(), gaussian.position.y(), best_distance_squared, best_point_index);
		return gaussian.position.z() <= points[best_point_index].z + height_offset;
	}

	size_t size() const
	{
		return points.size();
	}
};

void appendGaussian(
	int index,
	const std::vector<Eigen::Vector3f>& positions,
	const std::vector<Eigen::Vector4f>& rotations,
	const std::vector<Eigen::Vector3f>& log_scales,
	const std::vector<float>& opacities,
	const std::vector<SHs>& shs,
	std::vector<Gaussian>& output)
{
	Gaussian gaussian;
	gaussian.position = positions[index];
	gaussian.rotation = rotations[index];
	gaussian.scale = log_scales[index].array().exp();
	gaussian.opacity = opacities[index];
	gaussian.shs = shs[index];
	output.emplace_back(gaussian);
}

void collectLOD(
	int node_index,
	int target_depth,
	const std::vector<Node>& nodes,
	const std::vector<Eigen::Vector3f>& positions,
	const std::vector<Eigen::Vector4f>& rotations,
	const std::vector<Eigen::Vector3f>& log_scales,
	const std::vector<float>& opacities,
	const std::vector<SHs>& shs,
	std::vector<Gaussian>& output)
{
	const Node& node = nodes[node_index];
	if (node.depth == 0) {
		for (int i = 0; i < node.count_leafs; i++)
			appendGaussian(node.start + i, positions, rotations, log_scales, opacities, shs, output);
		return;
	}
	if (node.depth <= target_depth && node.count_merged > 0) {
		for (int i = 0; i < node.count_merged; i++)
			appendGaussian(node.start + node.count_leafs + i, positions, rotations, log_scales, opacities, shs, output);
		return;
	}
	for (int i = 0; i < node.count_children; i++)
		collectLOD(node.start_children + i, target_depth, nodes, positions, rotations, log_scales, opacities, shs, output);
}

void recTraverse(ExplicitTreeNode* node, int& zerocount)
{
	if (node->depth == 0)
		zerocount++;
	if (node->children.size() > 0 && node->depth == 0)
		throw std::runtime_error("Leaf nodes should never have children!");

	for (auto c : node->children)
	{
		recTraverse(c, zerocount);
	}
}

int main(int argc, char* argv[])
{
	if (argc < 2 || argc > 5)
		throw std::runtime_error(
			"Usage: GaussianPlyLODGenerator <ply_file_path(.ply)> [max_merge_scale_m] "
			"[trajectories.csv] [ground_height_offset_m]");

	float max_merge_scale = 0.1f;
	if (argc >= 3)
		max_merge_scale = std::stof(argv[2]);
	if (!std::isfinite(max_merge_scale) || max_merge_scale <= 0)
		throw std::runtime_error("max_merge_scale_m must be finite and greater than zero");
	float ground_height_offset = 0.3f;
	if (argc >= 5)
		ground_height_offset = std::stof(argv[4]);
	if (!std::isfinite(ground_height_offset))
		throw std::runtime_error("ground_height_offset_m must be finite");

	uint32_t sh_degree = 0;
	std::vector<Gaussian> gaussians;
	try
	{
		sh_degree = Loader::loadPly(argv[1], gaussians, 0);
	}
	catch (const std::runtime_error&)
	{
		std::cout << "Could not load .ply. Attempt loading .bin\n";
		std::string filename(argv[1]);
		filename.pop_back();
		filename.pop_back();
		filename.pop_back();
		filename = filename + "bin";
		std::cout << filename << std::endl;
		sh_degree = Loader::loadBin(filename.c_str(), gaussians, 0);
	}

	std::cout << "Generating" << std::endl;

	PointbasedKdTreeGenerator generator;
	auto root = generator.generate(gaussians);

	std::cout << "Merging" << std::endl;

	std::cout << "Only merging Gaussians with max(scale) < " << max_merge_scale
		<< " m and center distance <= " << max_merge_scale << " m" << std::endl;
	std::unique_ptr<TrajectoryHeightLookup> trajectory;
	std::function<bool(const Gaussian&)> merge_predicate;
	if (argc >= 4)
	{
		trajectory = std::make_unique<TrajectoryHeightLookup>(argv[3]);
		const TrajectoryHeightLookup* trajectory_ptr = trajectory.get();
		merge_predicate = [trajectory_ptr, ground_height_offset](const Gaussian& gaussian) {
			return trajectory_ptr->isBelowTrajectory(gaussian, ground_height_offset);
		};
		std::cout << "Ground-only merging enabled with " << trajectory->size()
			<< " ego trajectory samples: Gaussian Z <= nearest XY ego Z + "
			<< ground_height_offset << " m" << std::endl;
	}
	ClusterMerger merger(max_merge_scale, std::move(merge_predicate));
	merger.merge(root, gaussians);

	std::cout << "Fixing rotations" << std::endl;
	RotationAligner::align(root, gaussians);

	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector4f> rotations;
	std::vector<Eigen::Vector3f> log_scales;
	std::vector<float> opacities;
	std::vector<SHs> shs;
	std::vector<Node> basenodes;
	std::vector<Box> boxes;
	Writer::makeHierarchy(gaussians, root, positions, rotations, log_scales, opacities, shs, basenodes, boxes);
	const size_t splats_count = gaussians.size();
	gaussians.clear();

	std::array<std::vector<Gaussian>, LOD_LEVELS> gaussianLODFiles;
	for (int depth = 1; depth < LOD_LEVELS; depth++)
		collectLOD(0, depth, basenodes, positions, rotations, log_scales, opacities, shs, gaussianLODFiles[depth]);

	std::filesystem::path input_filepath(argv[1]);
	// write lod ply files.
	std::string filename_without_ext = input_filepath.stem().string();
	std::array<std::string, LOD_LEVELS> filenameLODs;
	for (int i = 0; i < LOD_LEVELS; i++) {
		if (i == 0) {
			filenameLODs[i] = input_filepath.filename().string();
		}
		else {
			filenameLODs[i] = filename_without_ext + "_LOD" + std::to_string(i) + ".ply";
			auto filepathLOD = input_filepath.parent_path() / filenameLODs[i];
			Writer::writePly(filepathLOD.string().c_str(), gaussianLODFiles[i], sh_degree);
		}
	}
	
	// write meta files.
	std::filesystem::path output_filepath = input_filepath;
	output_filepath.replace_extension(".3dgs");
	std::cout << "writing " << output_filepath.string() << std::endl;
	std::ofstream outfile(output_filepath.string());
	if (!outfile.good())
		throw std::runtime_error("File not created!");
	outfile << "{" << std::endl;
	outfile << "\t\"version\": \"1.0\"," << std::endl;
	outfile << "\t\"name\": \"" << output_filepath.filename().string() << "\"," << std::endl;
	outfile << "\t\"source\": \"" << input_filepath.filename().string() << "\"," << std::endl;
	outfile << "\t\"description\": \"Gaussian Splatting meta file with LOD definition.\"," << std::endl;
	outfile << "\t\"shDegree\": " << sh_degree << "," << std::endl;
	outfile << "\t\"splatsCount\": " << splats_count << "," << std::endl;
	outfile << "\t\"splatsLODFiles\": [";
	for (int i = 0; i < LOD_LEVELS; i++) {
		outfile << "\"" << filenameLODs[i] << "\"";
		if (i != LOD_LEVELS - 1) outfile << ",";
	}
	outfile << "]," << std::endl;
	outfile << "\t\"boundingBox\": {" << std::endl;
	outfile << "\t\t\"min\": [" << root->bounds.minn.x() <<", " << root->bounds.minn.y() << ", " << root->bounds.minn.z() << "]," << std::endl;
	outfile << "\t\t\"max\": [" << root->bounds.maxx.x() << ", " << root->bounds.maxx.y() << ", " << root->bounds.maxx.z() << "]" << std::endl;
	outfile << "\t}" << std::endl;
	outfile << "}" << std::endl;
	outfile.close();
}