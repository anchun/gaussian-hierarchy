#include "hierarchy_loader.h"
#include "hierarchy_explicit_loader.h"
#include "hierarchy_writer.h"
#include "writer.h"
#include "loader.h"
#include "merger_policy.h"
#include "dependencies/json.hpp"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace
{
void require(bool condition, const std::string& message)
{
	if (!condition)
		throw std::runtime_error(message);
}

struct TestFiles
{
	fs::path directory = "test_spatial_hierarchy_data";
	TestFiles()
	{
		require(!fs::exists(directory), "test directory already exists");
		fs::create_directory(directory);
	}
	~TestFiles() { fs::remove_all(directory); }
};

Gaussian gaussian(float x)
{
	Gaussian result;
	result.position = Eigen::Vector3f(x, 0.f, 10.f);
	result.scale = Eigen::Vector3f::Constant(0.05f);
	result.rotation = Eigen::Vector4f(1.f, 0.f, 0.f, 0.f);
	result.opacity = 0.75f;
	result.shs.setZero();
	computeCovariance(result.scale, result.rotation, result.covariance);
	return result;
}

ExplicitTreeNode leaf(int index, const std::vector<Gaussian>& gaussians)
{
	ExplicitTreeNode node;
	node.depth = 0;
	node.leaf_indices = {index};
	const Eigen::Vector3f delta = Eigen::Vector3f::Constant(0.2f);
	node.bounds = Box(gaussians[index].position - delta, gaussians[index].position + delta);
	return node;
}

ExplicitTreeNode branch(int depth, std::vector<ExplicitTreeNode*> children, const Gaussian& representative)
{
	ExplicitTreeNode node;
	node.depth = depth;
	node.children = std::move(children);
	node.merged.push_back(representative);
	node.bounds = node.children.front()->bounds;
	for (const auto* child : node.children)
	{
		node.bounds.minn.head<3>() = node.bounds.minn.head<3>().cwiseMin(child->bounds.minn.head<3>());
		node.bounds.maxx.head<3>() = node.bounds.maxx.head<3>().cwiseMax(child->bounds.maxx.head<3>());
	}
	return node;
}

void deleteChildren(ExplicitTreeNode& root)
{
	for (auto* child : root.children)
	{
		deleteChildren(*child);
		delete child;
	}
	root.children.clear();
}

struct Hierarchy
{
	std::vector<Eigen::Vector3f> positions, scales;
	std::vector<Eigen::Vector4f> rotations;
	std::vector<SHs> shs;
	std::vector<float> alphas;
	std::vector<Node> nodes;
	std::vector<Box> boxes;

	void load(const fs::path& path)
	{
		HierarchyLoader::load(path.string().c_str(), positions, shs, alphas, scales, rotations, nodes, boxes);
	}

	void write(const fs::path& path, bool compressed)
	{
		HierarchyWriter writer;
		writer.write(path.string().c_str(), positions.size(), nodes.size(), positions.data(),
			shs.data(), alphas.data(), scales.data(), rotations.data(), nodes.data(), boxes.data(), compressed);
	}
};

void checkCoverage(const Hierarchy& hierarchy, const std::vector<int>& frontier, int expected_leaves)
{
	std::vector<int> counts(hierarchy.positions.size(), 0);
	std::vector<int> pending = frontier;
	while (!pending.empty())
	{
		const Node& node = hierarchy.nodes.at(pending.back());
		pending.pop_back();
		for (int i = 0; i < node.count_leafs; ++i)
			++counts.at(node.start + i);
		for (int i = 0; i < node.count_children; ++i)
			pending.push_back(node.start_children + i);
	}
	int leaves = 0;
	for (const Node& node : hierarchy.nodes)
		for (int i = 0; i < node.count_leafs; ++i)
		{
			require(counts.at(node.start + i) == 1, "frontier missed or multiply covered a Gaussian leaf");
			++leaves;
		}
	require(leaves == expected_leaves, "unexpected Gaussian leaf count");
}

void writeJson(const fs::path& path, const json& value)
{
	fs::create_directories(path.parent_path());
	std::ofstream(path) << value.dump(2);
}

std::string quoted(const fs::path& path)
{
	return "\"" + path.string() + "\"";
}
}

int main()
{
	TestFiles files;
	std::vector<Gaussian> source{gaussian(-10), gaussian(0), gaussian(8), gaussian(10)};
	ExplicitTreeNode a = leaf(0, source), b = leaf(1, source), c = leaf(2, source), d = leaf(3, source);
	ExplicitTreeNode deep = branch(1, {&c, &d}, gaussian(9));
	ExplicitTreeNode middle = branch(2, {&b, &deep}, gaussian(4.5f));
	ExplicitTreeNode root = branch(3, {&a, &middle}, gaussian(-2.75f));
	std::string error;
	std::vector<int> ids;
	std::vector<Box> boxes;
	std::vector<Eigen::Vector3f> centers{Eigen::Vector3f::Constant(100000.f), Eigen::Vector3f::Zero()};
	for (const bool compressed : {false, true})
	{
		const fs::path path = files.directory / (compressed ? "compressed.hier" : "uncompressed.hier");
		Writer::writeHierarchy(path.string().c_str(), source, &root, compressed);
		Hierarchy hierarchy;
		hierarchy.load(path);
		require(HierarchyLoader::loadSpatialFrontier(path.string().c_str(), 3.f, ids, boxes, error), error);
		require(ids == std::vector<int>({1, 3, 4}), "adaptive frontier did not include shallow leaves");
		checkCoverage(hierarchy, ids, 4);
		require(HierarchyLoader::loadSpatialFrontier(path.string().c_str(), 1.f, ids, boxes, error), error);
		require(ids == std::vector<int>({1, 3, 5, 6}), "fine frontier is not complete");
		checkCoverage(hierarchy, ids, 4);
		require(HierarchyLoader::loadSpatialFrontier(path.string().c_str(), 100.f, ids, boxes, error)
			&& ids == std::vector<int>({0}), "coarse root frontier failed");
		checkCoverage(hierarchy, ids, 4);

		std::unordered_set<int> none;
		std::vector<Gaussian> loaded;
		ExplicitTreeNode loaded_root;
		require(HierarchyExplicitLoader::loadExplicit(path.string().c_str(), loaded, &loaded_root,
			0, centers, &none, false, true), "unmatched spatial source was dropped");
		require(loaded.size() == source.size(), "unmatched Gaussian retention is not 100%");
		for (std::size_t i = 0; i < source.size(); ++i)
			require(loaded[i].position == source[i].position && loaded[i].opacity == source[i].opacity,
				"spatial load mutated original leaf positions or attenuated opacity");
		deleteChildren(loaded_root);

		loaded.clear();
		loaded_root = ExplicitTreeNode{};
		const std::unordered_set<int> excluded{1, 3};
		require(HierarchyExplicitLoader::loadExplicit(path.string().c_str(), loaded, &loaded_root,
			0, centers, &excluded, false, true), "retained subtree was dropped");
		require(loaded.size() == 2 && loaded[0].position.x() == 8.f && loaded[1].position.x() == 10.f,
			"exclusion did not retain exactly the desired leaves");
		require(loaded_root.merged.size() == 1 && std::abs(loaded_root.merged[0].position.x() - 9.f) < 1e-6f,
			"ancestor representative was not rebuilt from retained children");
		require(loaded_root.bounds.minn.x() > 7.f, "excluded bounds contaminated the rebuilt spatial hierarchy");
		const fs::path rebuilt = files.directory / "rebuilt.hier";
		Writer::writeHierarchy(rebuilt.string().c_str(), loaded, &loaded_root, compressed);
		Hierarchy rebuilt_hierarchy;
		rebuilt_hierarchy.load(rebuilt);
		require(HierarchyLoader::loadSpatialFrontier(rebuilt.string().c_str(), 1.f, ids, boxes, error), error);
		checkCoverage(rebuilt_hierarchy, ids, 2);
		deleteChildren(loaded_root);

		loaded.clear();
		loaded_root = ExplicitTreeNode{};
		const std::unordered_set<int> all_excluded{0};
		require(!HierarchyExplicitLoader::loadExplicit(path.string().c_str(), loaded, &loaded_root,
			0, centers, &all_excluded, false, true) && loaded.empty(),
			"legitimate fully duplicated contribution was not empty");
		const fs::path truncated = files.directory / "truncated.hier";
		fs::copy_file(path, truncated, fs::copy_options::overwrite_existing);
		fs::resize_file(truncated, fs::file_size(truncated) - 1);
		require(!HierarchyLoader::loadSpatialFrontier(truncated.string().c_str(), 1.f, ids, boxes, error),
			"truncated hierarchy topology accepted");
		bool rejected = false;
		try { rebuilt_hierarchy.load(truncated); } catch (const std::exception&) { rejected = true; }
		require(rejected, "full loader accepted a truncated hierarchy");
		hierarchy.nodes[0].start_children = 0;
		hierarchy.write(files.directory / "cycle.hier", compressed);
		require(!HierarchyLoader::loadSpatialFrontier((files.directory / "cycle.hier").string().c_str(),
			1.f, ids, boxes, error), "invalid hierarchy topology accepted");
	}

	for (const bool compressed : {false, true})
	{
		const fs::path single = files.directory / "single.hier";
		Writer::writeHierarchy(single.string().c_str(), source, &a, compressed);
		require(HierarchyLoader::loadSpatialFrontier(single.string().c_str(), 1.f, ids, boxes, error)
			&& ids == std::vector<int>({0}), "single-leaf frontier failed");
		std::vector<Gaussian> loaded;
		ExplicitTreeNode loaded_root;
		require(HierarchyExplicitLoader::loadExplicit(single.string().c_str(), loaded, &loaded_root,
			0, centers, nullptr, false, true), "single leaf failed to load");
		require(loaded.size() == 1 && loaded[0].position == source[0].position,
			"single leaf moved to the chunk center");
		std::vector<Eigen::Vector3f> legacy_center{centers[0]};
		std::vector<Gaussian> legacy_loaded;
		ExplicitTreeNode legacy_root;
		require(HierarchyExplicitLoader::loadExplicit(single.string().c_str(), legacy_loaded, &legacy_root,
			0, legacy_center) && legacy_loaded[0].position == source[0].position,
			"single legacy leaf moved to the chunk center");
		std::vector<Gaussian> distant{gaussian(100000.f)};
		ExplicitTreeNode distant_leaf = leaf(0, distant);
		const fs::path distant_path = files.directory / "distant.hier";
		Writer::writeHierarchy(distant_path.string().c_str(), distant, &distant_leaf, compressed);
		require(HierarchyLoader::loadSpatialFrontier(distant_path.string().c_str(), 1.f, ids, boxes, error)
			&& ids == std::vector<int>({0}), "distant or half-overflowed boxes lost frontier coverage");
		loaded.clear();
		loaded_root = ExplicitTreeNode{};
		require(HierarchyExplicitLoader::loadExplicit(distant_path.string().c_str(), loaded, &loaded_root,
			0, centers, nullptr, false, true)
			&& loaded[0].position == distant[0].position && loaded[0].opacity == distant[0].opacity,
			"distant geometry was attenuated or moved");
	}
	ExplicitTreeNode mixed = root;
	mixed.leaf_indices = {0};
	mixed.children.clear();
	mixed.children.push_back(&middle);
	const fs::path mixed_path = files.directory / "mixed_direct_leaves.hier";
	Writer::writeHierarchy(mixed_path.string().c_str(), source, &mixed, false);
	Hierarchy mixed_hierarchy;
	mixed_hierarchy.load(mixed_path);
	require(HierarchyLoader::loadSpatialFrontier(mixed_path.string().c_str(), 1.f, ids, boxes, error)
		&& ids == std::vector<int>({0}), "direct leaves on internal nodes were omitted");
	checkCoverage(mixed_hierarchy, ids, 4);
	std::vector<Gaussian> mixed_loaded;
	ExplicitTreeNode mixed_loaded_root;
	require(HierarchyExplicitLoader::loadExplicit(mixed_path.string().c_str(), mixed_loaded, &mixed_loaded_root,
		0, centers, nullptr, false, true) && mixed_loaded.size() == 4, "direct internal leaves were dropped");
	deleteChildren(mixed_loaded_root);

	const fs::path empty = files.directory / "empty.hier";
	{
		std::ofstream file(empty, std::ios::binary);
		int zero = 0;
		file.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
	}
	require(!HierarchyLoader::loadSpatialFrontier(empty.string().c_str(), 1.f, ids, boxes, error),
		"empty source hierarchy accepted");
	bool empty_rejected = false;
	try
	{
		std::vector<Gaussian> loaded;
		ExplicitTreeNode loaded_root;
		const std::unordered_set<int> excluded{0};
		HierarchyExplicitLoader::loadExplicit(empty.string().c_str(), loaded, &loaded_root,
			0, centers, &excluded, false, true);
	}
	catch (const std::exception&) { empty_rejected = true; }
	require(empty_rejected, "empty source was confused with a fully excluded contribution");
	ExplicitTreeNode representative_only;
	representative_only.depth = 0;
	representative_only.merged.push_back(source[0]);
	representative_only.bounds = a.bounds;
	const fs::path no_leaves = files.directory / "no_leaves.hier";
	Writer::writeHierarchy(no_leaves.string().c_str(), source, &representative_only, true);
	empty_rejected = false;
	try
	{
		std::vector<Gaussian> loaded;
		ExplicitTreeNode loaded_root;
		const std::unordered_set<int> excluded{0};
		HierarchyExplicitLoader::loadExplicit(no_leaves.string().c_str(), loaded, &loaded_root,
			0, centers, &excluded, false, true);
	}
	catch (const std::exception&) { empty_rejected = true; }
	require(empty_rejected, "a source without actual leaves was treated as a duplicate contribution");

	// Exercise the actual merger entry point in both output modes with fully excluded chunks,
	// then again with no matches and intentionally misleading legacy centers.
	std::ifstream fixture_file(SPATIAL_FIXTURE_PATH);
	json fixture;
	fixture_file >> fixture;
	const fs::path sources = files.directory / "sources";
	const fs::path trained = files.directory / "trained" / "chunks";
	const fs::path scaffold = trained.parent_path() / "scaffold" / "point_cloud" / "iteration_30000";
	fs::create_directories(scaffold);
	Writer::writePly((scaffold / "point_cloud.ply").string().c_str(), {gaussian(100)}, 3);
	std::ofstream(scaffold / "pc_info.txt") << "0\n";
	writeJson(scaffold / "prior_info.json", {{"road_points", 0}, {"skybox_points", 0}, {"fixed_prior_points", 0}});
	for (int chunk = 0; chunk < 3; ++chunk)
	{
		const std::string name = std::to_string(chunk) + "_0";
		json document = fixture;
		document["chunk_id"] = chunk;
		document["core_frame_names"] = {std::to_string(chunk) + ".jpg"};
		writeJson(sources / name / "chunk_info.json", document);
		writeJson(trained / name / "chunk_info.json", document);
		std::ofstream(sources / name / "center.txt") << "100000 100000 100000\n";
		writeJson(trained / name / "cameras.json", json::array({{
			{"img_name", "front/" + std::to_string(chunk) + ".jpg"}, {"position", {0, 0, 0}},
			{"rotation", {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
			{"fx", 1000}, {"fy", 1000}, {"width", 1000}, {"height", 1000}
		}}));
	}
	for (const bool duplicates : {true, false})
	{
		std::vector<float> expected;
		for (int chunk = 0; chunk < 3; ++chunk)
		{
			const std::string name = std::to_string(chunk) + "_0";
			std::vector<Gaussian> gaussians{gaussian(duplicates ? -5.f : -5.f - 5.f * chunk)};
			ExplicitTreeNode single = leaf(0, gaussians);
			Writer::writeHierarchy((trained / name / "hierarchy.hier").string().c_str(),
				gaussians, &single, chunk != 1);
			if (!duplicates || chunk == 0)
				expected.push_back(gaussians[0].position.x());
		}
		std::sort(expected.begin(), expected.end());
		for (const bool ply : {false, true})
		{
			const fs::path output = files.directory / (ply ? "merged.ply" : "merged.hier");
			const std::string command = quoted(SPATIAL_MERGER_PATH) + " " + quoted(trained) + " 3 "
				+ quoted(sources) + " " + quoted(output) + " 2_0 1_0 0_0";
			require(std::system(command.c_str()) == 0, "spatial merger entry point failed");
			std::vector<float> actual;
			if (ply)
			{
				std::vector<Gaussian> merged;
				Loader::loadPly(output.string().c_str(), merged);
				for (const Gaussian& g : merged)
				{
					actual.push_back(g.position.x());
					require(std::abs(g.opacity - 0.75f) < 1e-6f, "PLY spatial opacity was attenuated");
				}
			}
			else
			{
				Hierarchy merged;
				merged.load(output);
				require(HierarchyLoader::loadSpatialFrontier(output.string().c_str(), 1.f, ids, boxes, error), error);
				checkCoverage(merged, ids, expected.size());
				for (const Node& node : merged.nodes)
					for (int leaf = 0; leaf < node.count_leafs; ++leaf)
					{
						actual.push_back(merged.positions[node.start + leaf].x());
						require(merged.alphas[node.start + leaf] == 0.75f, "hierarchy spatial opacity was attenuated");
					}
			}
			std::sort(actual.begin(), actual.end());
			require(actual == expected, "PLY/hierarchy outputs disagree on retained Gaussian leaves");
		}
	}
}
