
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>

// Handy typedefs.
typedef pcl::FPFHSignature33 FPFHDescriptor;
typedef pcl::PFHSignature125 PFHDescriptor;

class PCLWrapperFPFH {
public:
	const float voxelSize = 0.01f;
	const float tsdfThreshOccupied = voxelSize;	//thres for voxel considered to be occupied;

	// not necessarily symmetric
    float computeDistanceFPFH(const Grid3f& tsdfA, const Grid3f& tsdfB) {
		pcl::PointCloud<FPFHDescriptor>::Ptr descriptorsA(new pcl::PointCloud<FPFHDescriptor>());
        pcl::PointCloud<FPFHDescriptor>::Ptr descriptorsB(new pcl::PointCloud<FPFHDescriptor>());

		unsigned int idxA = computeFPFHDescriptor(tsdfA, descriptorsA);
		unsigned int idxB = computeFPFHDescriptor(tsdfB, descriptorsB);

        // PCL crashes when called on an empty cloud
        if (idxA == 0xFFFFFFFF || idxB == 0xFFFFFFFF)
            return 0.0f;

		float sqrtDist = 0.0f;
		for (unsigned int i = 0; i < descriptorsA->points[idxA].descriptorSize(); i++) {
			float d = descriptorsA->points[idxA].histogram[i] - descriptorsB->points[idxB].histogram[i];
			sqrtDist += d*d;
		}
		return sqrtDist;
	}

    // not necessarily symmetric
    float computeDistancePFH(const Grid3f& tsdfA, const Grid3f& tsdfB) {
        pcl::PointCloud<PFHDescriptor>::Ptr descriptorsA(new pcl::PointCloud<PFHDescriptor>());
        pcl::PointCloud<PFHDescriptor>::Ptr descriptorsB(new pcl::PointCloud<PFHDescriptor>());

        unsigned int idxA = computePFHDescriptor(tsdfA, descriptorsA);
        unsigned int idxB = computePFHDescriptor(tsdfB, descriptorsB);

        // PCL crashes when called on an empty cloud
        if (idxA == 0xFFFFFFFF || idxB == 0xFFFFFFFF)
            return 0.0f;

        float sqrtDist = 0.0f;
        for (unsigned int i = 0; i < descriptorsA->points[idxA].descriptorSize(); i++) {
            float d = descriptorsA->points[idxA].histogram[i] - descriptorsB->points[idxB].histogram[i];
            sqrtDist += d*d;
        }
        return sqrtDist;
    }

    void makeCloudNormals(const Grid3f& tsdf, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::Normal>::Ptr &normals)
    {
        const bool usePCANormals = true;
        const float PCANormalRadius = 20.0f;
        const float threshold = voxelSize * 2.0f;

        for (size_t z = 0; z < tsdf.getDimZ(); z++) {
            for (size_t y = 0; y < tsdf.getDimY(); y++) {
                for (size_t x = 0; x < tsdf.getDimX(); x++) {
                    if (std::abs(tsdf(x, y, z)) < threshold) {
                        vec3f p = vec3f((float)x, (float)y, (float)z);
                        vec3f n;
                        if (computeSurfaceNormal(x, y, z, n, tsdf)) {

                            p -= n * tsdf(x, y, z);
                            vec3f iNormal;
                            if (getSurfaceNormal(tsdf, p, iNormal)) {
                                pcl::PointXYZ pPCL;
                                pPCL.x = p.x;
                                pPCL.y = p.y;
                                pPCL.z = p.z;
                                cloud->push_back(pPCL);

                                pcl::Normal nPCL;
                                nPCL.normal_x = n.x;
                                nPCL.normal_y = n.y;
                                nPCL.normal_z = n.z;
                                normals->push_back(nPCL);
                            }
                        }
                    }
                }
            }
        }

        if (usePCANormals)
        {
            normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);

            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEstimator;
            pcl::search::KdTree<pcl::PointXYZ>::Ptr searchMethod(new pcl::search::KdTree<pcl::PointXYZ>);
            normEstimator.setInputCloud(cloud);
            normEstimator.setSearchMethod(searchMethod);
            normEstimator.setRadiusSearch(PCANormalRadius);
            normEstimator.compute(*normals);
        }
    }
	
    unsigned int computeFPFHDescriptor(const Grid3f& tsdf, pcl::PointCloud<FPFHDescriptor>::Ptr& descriptors) {

		const unsigned int dim = tsdf.getDimX();

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

        makeCloudNormals(tsdf, cloud, normals);

        if (cloud->size() == 0)
        {
            return 0xFFFFFFFF;
        }

		pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
		kdtree->setInputCloud(cloud);
		pcl::PointXYZ queryPoint((dim - 1) / 2, (dim - 1) / 2, (dim - 1) / 2);
		std::vector<int> pointIdxNKNSearch(1);
		std::vector<float> pointNKNSquaredDistance(1);
		int n = kdtree->nearestKSearch(queryPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
		assert(n);

        // FPFH estimation object.
        pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
        fpfh.setInputCloud(cloud);
        fpfh.setInputNormals(normals);
        fpfh.setSearchMethod(kdtree);
        // Search radius, to look for neighbors. Note: the value given here has to be
        // larger than the radius used to estimate the normals.
        fpfh.setRadiusSearch(100.0f);

        fpfh.compute(*descriptors);

		return pointIdxNKNSearch[0];
	}

    unsigned int computePFHDescriptor(const Grid3f& tsdf, pcl::PointCloud<PFHDescriptor>::Ptr& descriptors) {

        const unsigned int dim = tsdf.getDimX();

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

        makeCloudNormals(tsdf, cloud, normals);

        if (cloud->size() == 0)
        {
            return 0xFFFFFFFF;
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        kdtree->setInputCloud(cloud);
        pcl::PointXYZ queryPoint((dim - 1) / 2, (dim - 1) / 2, (dim - 1) / 2);
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);
        int n = kdtree->nearestKSearch(queryPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
        assert(n);

        // FPFH estimation object.
        pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
        pfh.setInputCloud(cloud);
        pfh.setInputNormals(normals);
        pfh.setSearchMethod(kdtree);
        // Search radius, to look for neighbors. Note: the value given here has to be
        // larger than the radius used to estimate the normals.
        pfh.setRadiusSearch(10.0f);

        pfh.compute(*descriptors);

        return pointIdxNKNSearch[0];
    }

private:
    static float frac(float val) {
        return (val - floorf(val));
    }
    static vec3f frac(const vec3f& val) {
        return vec3f(frac(val.x), frac(val.y), frac(val.z));
    }

    // voxelPos should be in (0, 0, 0) to (31, 31, 31) space
    static float getTSDFCell(const Grid3f &tsdf, const vec3f& voxelPosFloat)
    {
        vec3i voxelPos = math::round(voxelPosFloat);
        if (tsdf.isValidCoordinate(voxelPos.x, voxelPos.y, voxelPos.z))
            return tsdf(voxelPos.x, voxelPos.y, voxelPos.z);
        else
            return std::numeric_limits<float>::infinity();
    }

    bool trilinearInterpolation(const Grid3f &tsdf, const vec3f& pos, float& result) const {
        const float oSet = 1.0f;
        const vec3f posDual = pos - vec3f(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f);
        vec3f weight = frac(posDual);

        float dist = 0.0f;
        result = 0.0f;

        float v;
        v = getTSDFCell(tsdf, posDual + vec3f(0.0f, 0.0f, 0.0f)); if(v > 1000.0f) return false; dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*v; result += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*v;
        v = getTSDFCell(tsdf, posDual + vec3f(oSet, 0.0f, 0.0f)); if(v > 1000.0f) return false; dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*v; result += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*v;
        v = getTSDFCell(tsdf, posDual + vec3f(0.0f, oSet, 0.0f)); if(v > 1000.0f) return false; dist += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*v; result += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*v;
        v = getTSDFCell(tsdf, posDual + vec3f(0.0f, 0.0f, oSet)); if(v > 1000.0f) return false; dist += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *v; result += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *v;
        v = getTSDFCell(tsdf, posDual + vec3f(oSet, oSet, 0.0f)); if(v > 1000.0f) return false; dist += weight.x *	   weight.y *(1.0f - weight.z)*v; result += weight.x *	   weight.y *(1.0f - weight.z)*v;
        v = getTSDFCell(tsdf, posDual + vec3f(0.0f, oSet, oSet)); if(v > 1000.0f) return false; dist += (1.0f - weight.x)*	   weight.y *	   weight.z *v; result += (1.0f - weight.x)*	   weight.y *	   weight.z *v;
        v = getTSDFCell(tsdf, posDual + vec3f(oSet, 0.0f, oSet)); if(v > 1000.0f) return false; dist += weight.x *(1.0f - weight.y)*	   weight.z *v; result += weight.x *(1.0f - weight.y)*	   weight.z *v;
        v = getTSDFCell(tsdf, posDual + vec3f(oSet, oSet, oSet)); if(v > 1000.0f) return false; dist += weight.x *	   weight.y *	   weight.z *v; result += weight.x *	   weight.y *	   weight.z *v;

        return true;
    }

    bool getSurfaceNormal(const Grid3f& tsdf, const vec3f &pos, vec3f &result) const {
        vec3f n0, n1;
        if (!trilinearInterpolation(tsdf, pos + vec3f( 1.0f, 0.0f, 0.0f), n1.x)) return false;
        if (!trilinearInterpolation(tsdf, pos + vec3f(-1.0f, 0.0f, 0.0f), n0.x)) return false;
        if (!trilinearInterpolation(tsdf, pos + vec3f(0.0f,  1.0f, 0.0f), n1.y)) return false;
        if (!trilinearInterpolation(tsdf, pos + vec3f(0.0f, -1.0f, 0.0f), n0.y)) return false;
        if (!trilinearInterpolation(tsdf, pos + vec3f(0.0f, 0.0f,  1.0f), n1.z)) return false;
        if (!trilinearInterpolation(tsdf, pos + vec3f(0.0f, 0.0f, -1.0f), n0.z)) return false;

        result = (n1 - n0).getNormalized();
        return true;
    }

    bool computeSurfaceNormalFloat(float x, float y, float z, vec3f& res, const Grid3f& tsdf) {
        if (!tsdf.isValidCoordinate(x, y, z))	return false;
        if (!tsdf.isValidCoordinate(x + 1, y, z))	return false;
        if (!tsdf.isValidCoordinate(x, y + 1, z))	return false;
        if (!tsdf.isValidCoordinate(x, y, z + 1))	return false;
        if (!tsdf.isValidCoordinate(x - 1, y, z))	return false;
        if (!tsdf.isValidCoordinate(x, y - 1, z))	return false;
        if (!tsdf.isValidCoordinate(x, y, z - 1))	return false;

        float SDFx = tsdf(x + 1, y, z) - tsdf(x - 1, y, z);
        float SDFy = tsdf(x, y + 1, z) - tsdf(x, y - 1, z);
        float SDFz = tsdf(x, y, z + 1) - tsdf(x, y, z - 1);
        if (SDFx == 0 && SDFy == 0 && SDFz == 0) // Don't divide by zero!
            return false;
        else {
            res = vec3f(SDFx, SDFy, SDFz).getNormalized();
            return true;
        }
    }

	bool computeSurfaceNormal(int x, int y, int z, vec3f& res, const Grid3f& tsdf) {
		if (!tsdf.isValidCoordinate(x, y, z))	return false;
		if (!tsdf.isValidCoordinate(x + 1, y, z))	return false;
		if (!tsdf.isValidCoordinate(x, y + 1, z))	return false;
		if (!tsdf.isValidCoordinate(x, y, z + 1))	return false;
		if (!tsdf.isValidCoordinate(x - 1, y, z))	return false;
		if (!tsdf.isValidCoordinate(x, y - 1, z))	return false;
		if (!tsdf.isValidCoordinate(x, y, z - 1))	return false;

		float SDFx = tsdf(x + 1, y, z) - tsdf(x - 1, y, z);
		float SDFy = tsdf(x, y + 1, z) - tsdf(x, y - 1, z);
		float SDFz = tsdf(x, y, z + 1) - tsdf(x, y, z - 1);
		if (SDFx == 0 && SDFy == 0 && SDFz == 0) // Don't divide by zero!
			return false;
		else {
			res = vec3f(SDFx, SDFy, SDFz).getNormalized();
			return true;
		}
	}

};
