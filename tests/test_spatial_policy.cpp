#include "merger_policy.h"
#include "dependencies/json.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace
{
void require(bool condition, const std::string& message)
{
	if (!condition)
		throw std::runtime_error(message);
}

void writeJson(const fs::path& path, const json& value)
{
	fs::create_directories(path.parent_path());
	std::ofstream file(path);
	file << value.dump(2);
	require(file.good(), "failed to write test metadata");
}

OwnershipUnit unit(int id, float x, float y = 0.f, float z = 10.f, float width = 0.4f)
{
	const Eigen::Vector3f center(x, y, z);
	const Eigen::Vector3f half = Eigen::Vector3f::Constant(width / 2.f);
	return {id, Box(center - half, center + half)};
}

struct TestFiles
{
	fs::path directory = "test_spatial_policy_data";
	TestFiles()
	{
		require(!fs::exists(directory), "test directory already exists");
		fs::create_directory(directory);
	}
	~TestFiles() { fs::remove_all(directory); }
};
}

int main()
{
	TestFiles files;
	std::ifstream fixture_file(SPATIAL_FIXTURE_PATH);
	json fixture;
	fixture_file >> fixture;
	std::string error;
	const fs::path path = files.directory / "chunk_info.json";
	writeJson(path, fixture);
	SpatialChunkInfo info;
	require(loadSpatialChunkInfo(path.string(), info, error), error);
	for (const json& item : fixture.at("owner_cases"))
	{
		Eigen::Vector2d xy(item["xy"][0].get<double>(), item["xy"][1].get<double>());
		require(spatialOwner(info, xy) == item["owner"].get<int>(), "shared Python/C++ ownership boundary differs");
	}
	require(spatialOwner(info, Eigen::Vector2d(std::nextafter(0., -1.), 10.)) == 0,
		"double-precision split boundary was rounded");
	bool rejected_position = false;
	try { spatialOwner(info, Eigen::Vector2d(std::numeric_limits<double>::infinity(), 0.)); }
	catch (const std::exception&) { rejected_position = true; }
	require(rejected_position, "non-finite owner position accepted");

	std::vector<std::function<void(json&)>> malformed{
		[](json& j) { j.erase("partition_mode"); },
		[](json& j) { j["partition_mode"] = "unknown_spatial"; },
		[](json& j) { j["schema_version"] = 2; },
		[](json& j) { j["schema_version"] = true; },
		[](json& j) { j["schema_version"] = 1.0; },
		[](json& j) { j["partition_id"] = "bad"; },
		[](json& j) { j.erase("selection_id"); },
		[](json& j) { j["chunk_id"] = 3; },
		[](json& j) { j["chunk_count"] = 0; },
		[](json& j) { j["overlap_m"] = -1; },
		[](json& j) { j["overlap_m"] = true; },
		[](json& j) { j["context_distance_m"] = 4; },
		[](json& j) { j["context_frame_stride"] = 0; },
		[](json& j) { j["target_chunk_size_m"] = 0; },
		[](json& j) { j["core_frame_names"] = {"dir/frame.jpg"}; },
		[](json& j) { j["core_frame_names"] = {"same.jpg", "same.jpg"}; },
		[](json& j) { j["partition_tree"] = json::array(); },
		[](json& j) { j["partition_tree"][0]["axis"] = 2; },
		[](json& j) { j["partition_tree"][0]["cut"] = nullptr; },
		[](json& j) { j["partition_tree"][0]["left"] = 0; },
		[](json& j) { j["partition_tree"][0]["left"] = 10; },
		[](json& j) { j["partition_tree"][0]["right"] = 1; },
		[](json& j) { j["partition_tree"][2]["axis"] = 0; j["partition_tree"][2]["cut"] = -1.; },
		[](json& j) { j["partition_tree"][4]["chunk_id"] = 1; },
		[](json& j) { j["partition_tree"][1]["left"] = 0; }
	};
	for (std::size_t index = 0; index < malformed.size(); ++index)
	{
		json invalid = fixture;
		malformed[index](invalid);
		writeJson(path, invalid);
		error.clear();
		require(!loadSpatialChunkInfo(path.string(), info, error) && !error.empty(),
			"malformed metadata accepted: " + std::to_string(index));
	}
	std::ofstream(path) << "{\"partition_mode\":";
	require(!loadSpatialChunkInfo(path.string(), info, error), "truncated JSON accepted");

	std::vector<std::string> sources, trained;
	std::vector<json> documents;
	std::vector<SpatialChunkInfo> infos;
	bool spatial = false;
	for (int chunk = 0; chunk < 3; ++chunk)
	{
		json document = fixture;
		document["chunk_id"] = chunk;
		document["core_frame_names"] = {std::to_string(chunk) + ".jpg"};
		documents.push_back(document);
		sources.push_back((files.directory / ("source" + std::to_string(chunk))).string());
		trained.push_back((files.directory / ("trained" + std::to_string(chunk))).string());
		writeJson(fs::path(sources.back()) / "chunk_info.json", document);
		writeJson(fs::path(trained.back()) / "chunk_info.json", document);
	}
	require(validateSpatialChunkSet(sources, trained, infos, spatial, error) && spatial, error);
	require(!validateSpatialChunkSet({sources[0], sources[1]}, {trained[0], trained[1]}, infos, spatial, error),
		"incomplete chunk set accepted");
	require(!validateSpatialChunkSet({sources[0], sources[1], sources[1]},
		{trained[0], trained[1], trained[1]}, infos, spatial, error), "duplicate chunk ids accepted");
	for (const auto& mutate : std::vector<std::function<void(json&)>>{
		[](json& j) { j["partition_id"] = std::string(64, 'c'); },
		[](json& j) { j["partition_tree"][2]["cut"] = 11.; },
		[](json& j) { j["overlap_m"] = 6.; },
		[](json& j) { j["core_frame_names"] = {"0.jpg"}; }
	})
	{
		json changed = documents[2];
		mutate(changed);
		writeJson(fs::path(sources[2]) / "chunk_info.json", changed);
		writeJson(fs::path(trained[2]) / "chunk_info.json", changed);
		require(!validateSpatialChunkSet(sources, trained, infos, spatial, error),
			"conflicting partition set accepted");
	}
	writeJson(fs::path(sources[2]) / "chunk_info.json", documents[2]);
	writeJson(fs::path(trained[2]) / "chunk_info.json", documents[2]);
	json changed = documents[2];
	changed["final_camera_count"] = 17;
	writeJson(fs::path(trained[2]) / "chunk_info.json", changed);
	require(!validateSpatialTraining(sources[2], trained[2], info, error),
		"full-JSON snapshot mismatch accepted");
	fs::remove(fs::path(trained[2]) / "chunk_info.json");
	require(!validateSpatialTraining(sources[2], trained[2], info, error), "missing training snapshot accepted");
	writeJson(fs::path(trained[2]) / "chunk_info.json", documents[2]);
	writeJson(fs::path(sources[2]) / "chunk_info.json", {{"route_mode", "corridor"}});
	require(!validateSpatialTraining(sources[2], trained[2], info, error),
		"legacy source accepted a spatial training snapshot");
	writeJson(fs::path(trained[2]) / "chunk_info.json", {{"route_mode", "corridor"}});
	require(!validateSpatialChunkSet(sources, trained, infos, spatial, error), "mixed spatial/legacy set accepted");
	require(validateSpatialChunkSet({sources[2]}, {trained[2]}, infos, spatial, error) && !spatial,
		"legacy metadata no longer accepted");
	fs::remove(fs::path(sources[2]) / "chunk_info.json");
	fs::remove(fs::path(trained[2]) / "chunk_info.json");
	require(validateSpatialChunkSet({sources[2]}, {trained[2]}, infos, spatial, error) && !spatial,
		"absent legacy metadata no longer accepted");
	std::ofstream(fs::path(sources[2]) / "chunk_info.json") << "{\"route_mode\":";
	require(validateSpatialChunkSet({sources[2]}, {trained[2]}, infos, spatial, error) && !spatial,
		"unmarked legacy route-parser fallback behavior changed");
	writeJson(fs::path(sources[2]) / "chunk_info.json", documents[2]);
	writeJson(fs::path(trained[2]) / "chunk_info.json", documents[2]);
	require(validateSpatialChunkSet(sources, trained, infos, spatial, error), error);

	std::vector<std::vector<CameraSupport>> cameras(3);
	std::vector<std::vector<OwnershipUnit>> units{{unit(10, -5)}, {unit(20, -5)}, {unit(30, -5)}};
	SpatialOwnershipPlan plan = buildSpatialOwnership(units, infos, cameras);
	require(plan.stats.matched_pairs == 3 && plan.stats.excluded_units == 2,
		"three-way duplicates did not keep exactly one representative");
	require(plan.excluded_nodes[0].empty() && plan.excluded_nodes[1].count(20) && plan.excluded_nodes[2].count(30),
		"actual BSP core owner not preferred");
	for (auto& chunk : units)
		chunk.front().bounds = unit(0, 5, 20).bounds;
	plan = buildSpatialOwnership(units, infos, cameras);
	require(plan.excluded_nodes[2].empty() && plan.excluded_nodes[0].count(10) && plan.excluded_nodes[1].count(20),
		"third chunk core owner was removed");
	std::reverse(units.begin(), units.end());
	std::reverse(infos.begin(), infos.end());
	SpatialOwnershipPlan reversed = buildSpatialOwnership(units, infos, cameras);
	for (int chunk = 0; chunk < 3; ++chunk)
		require(reversed.excluded_nodes[2 - chunk] == plan.excluded_nodes[chunk],
			"chunk ordering changed spatial decisions");
	std::reverse(infos.begin(), infos.end());

	units = {{unit(10, -5)}, {unit(20, 5)}, {unit(30, 5, 20)}};
	plan = buildSpatialOwnership(units, infos, cameras);
	require(plan.stats.matched_pairs == 0 && plan.stats.excluded_units == 0
		&& plan.stats.unmatched_units == 3, "zero-match spatial units were removed");
	units = {{unit(10, -5)}, {unit(20, -5), unit(21, -20)}, {unit(30, 20)}};
	plan = buildSpatialOwnership(units, infos, cameras);
	require(plan.stats.unmatched_units == 2 && plan.stats.excluded_units == 1
		&& !plan.excluded_nodes[1].count(21) && !plan.excluded_nodes[2].count(30),
		"unmatched unit retention is not 100%");

	// A-B and B-C overlap >=90%, but A-C does not. An endpoint owner may remove B, never C.
	units = {{unit(10, -5)}, {unit(20, -4.97f)}, {unit(30, -4.94f)}};
	plan = buildSpatialOwnership(units, infos, cameras);
	require(plan.stats.matched_pairs == 2 && plan.stats.excluded_units == 1
		&& plan.excluded_nodes[1].count(20) && plan.excluded_nodes[2].empty(),
		"transitive chain incorrectly removed a nonmatching copy");

	units = {{unit(10, 5)}, {}, {unit(30, 5)}};
	plan = buildSpatialOwnership(units, infos, cameras);
	require(plan.excluded_nodes[0].empty() && plan.excluded_nodes[2].count(30),
		"missing core owner did not choose a real deterministic candidate");
	CameraSupport visible;
	visible.position = Eigen::Vector3f(5, 0, 0);
	visible.fx = visible.fy = 1000;
	visible.width = visible.height = 1000;
	visible.core = true;
	cameras[2].push_back(visible);
	plan = buildSpatialOwnership(units, infos, cameras);
	require(plan.excluded_nodes[0].count(10) && plan.excluded_nodes[2].empty(),
		"missing core owner did not use camera support");
	units = {{unit(10, 0)}, {unit(20, 0)}, {}};
	cameras.assign(3, {});
	cameras[0].push_back(visible);
	cameras[0][0].position.x() = 0;
	plan = buildSpatialOwnership(units, infos, cameras);
	require(plan.excluded_nodes[0].empty() && plan.excluded_nodes[1].count(20),
		"boundary camera arbitration failed");

	cameras.assign(3, {});
	units = {{unit(10, -5), unit(11, -5.01f)}, {unit(20, -5)}, {}};
	plan = buildSpatialOwnership(units, infos, cameras);
	require(plan.stats.matched_pairs == 1 && plan.stats.unmatched_units == 1
		&& !plan.excluded_nodes[0].count(11), "non-reciprocal unit was discarded");
	units = {{unit(10, -5, 0, 10, 100)}, {unit(20, -5, 0, 10, 100)}, {}};
	plan = buildSpatialOwnership(units, infos, cameras);
	require(plan.stats.uncertain_units == 2 && plan.stats.excluded_units == 0,
		"uncertain oversized units were discarded");
	units = {{unit(10, 100000)}, {unit(20, 100000)}, {}};
	units[0][0].bounds.minn.x() = units[0][0].bounds.maxx.x() = std::numeric_limits<float>::infinity();
	units[1][0].bounds = units[0][0].bounds;
	plan = buildSpatialOwnership(units, infos, cameras);
	require(plan.stats.uncertain_units == 2 && plan.stats.excluded_units == 0,
		"half-overflowed distant boxes were matched or discarded");

	const fs::path camera_path = files.directory / "cameras.json";
	writeJson(camera_path, json::array({{
		{"img_name", "front/0.jpg"}, {"position", {0, 0, 0}},
		{"rotation", {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
		{"fx", 1000}, {"fy", 1000}, {"width", 1000}, {"height", 1000}
	}}));
	std::vector<CameraSupport> support;
	require(loadSpatialCameraSupport(camera_path.string(), infos[0], support, error)
		&& support.size() == 1 && support[0].core, "spatial rig camera support classification failed");
}
