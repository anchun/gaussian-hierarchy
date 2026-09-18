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
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "appearance_filter.h"
#include "rotation_aligner.h"
#include "hierarchy_writer.h"

#define LOD_LEVELS 6
using json = nlohmann::json;

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
	if (argc < 2)
		throw std::runtime_error("Failed to pass args <ply_file_path(.ply)> [max_merge_scale_m]");

	float max_merge_scale = 0.1f;
	if (argc >= 3)
		max_merge_scale = std::stof(argv[2]);
	if (!std::isfinite(max_merge_scale) || max_merge_scale <= 0)
		throw std::runtime_error("max_merge_scale_m must be finite and greater than zero");

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
	ClusterMerger merger(max_merge_scale);
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