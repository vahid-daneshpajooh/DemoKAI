// FaceMeshKeypoints.h
#ifndef FACEMESHKEYPOINTS_H
#define FACEMESHKEYPOINTS_H

#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

/**
 * @brief maps feature keypoints to facial landmarks
 * @note  keypoints are 1-based: (1, 2, ..., 468)
 * @note  Iris keypoints (469-477) are not implemented in our current FFTFLite model
 */
struct FaceMeshTFLite {
    
    // a set of keys (facial landmarks)
    std::vector<std::string> dictKeys;

    // Map to hold the feature names and their corresponding keypoints
    std::unordered_map<std::string, std::vector<int>> dict_TFLite;

    // Constructor to initialize the map with landmark annotations
    FaceMeshTFLite() {
         dict_TFLite = {
            {"silhouette", {10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109} },
            {"lipsUpperOuter", {61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291}},
            {"lipsLowerOuter", {146, 91, 181, 84, 17, 314, 405, 321, 375, 291}},
            {"lipsUpperInner", {78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308}},
            {"lipsLowerInner", {78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308}},
            {"rightEyeUpper0", {246, 161, 160, 159, 158, 157, 173}},
            {"rightEyeLower0", {33, 7, 163, 144, 145, 153, 154, 155, 133}},
            {"rightEyeUpper1", {247, 30, 29, 27, 28, 56, 190}},
            {"rightEyeLower1", {130, 25, 110, 24, 23, 22, 26, 112, 243}},
            {"rightEyeUpper2", {113, 225, 224, 223, 222, 221, 189}},
            {"rightEyeLower2", {226, 31, 228, 229, 230, 231, 232, 233, 244}},
            {"rightEyeLower3", {143, 111, 117, 118, 119, 120, 121, 128, 245}},
            {"rightEyebrowUpper", {156, 70, 63, 105, 66, 107, 55, 193}},
            {"rightEyebrowLower", {35, 124, 46, 53, 52, 65}},
            {"rightEyeIris", {473, 474, 475, 476, 477}},
            {"leftEyeUpper0", {466, 388, 387, 386, 385, 384, 398}},
            {"leftEyeLower0", {263, 249, 390, 373, 374, 380, 381, 382, 362}},
            {"leftEyeUpper1", {467, 260, 259, 257, 258, 286, 414}},
            {"leftEyeLower1", {359, 255, 339, 254, 253, 252, 256, 341, 463}},
            {"leftEyeUpper2", {342, 445, 444, 443, 442, 441, 413}},
            {"leftEyeLower2", {446, 261, 448, 449, 450, 451, 452, 453, 464}},
            {"leftEyeLower3", {372, 340, 346, 347, 348, 349, 350, 357, 465}},
            {"leftEyebrowUpper", {383, 300, 293, 334, 296, 336, 285, 417}},
            {"leftEyebrowLower", {265, 353, 276, 283, 282, 295}},
            {"leftEyeIris", {468, 469, 470, 471, 472}},
            {"midwayBetweenEyes", {168}},
            {"noseTip", {1}},
            {"noseBottom", {2}},
            {"noseRightCorner", {98}},
            {"noseLeftCorner", {327}},
            {"rightCheek", {205}},
            {"leftCheek", {425}}
        };

        // TODO: create a list for "Dlib", and "Dlib-extended"
        // list of "All" keypoints
        dictKeys = {
            "silhouette",
            "lipsUpperOuter",
            "lipsLowerOuter",
            "lipsLowerInner",
            "rightEyeUpper0",
            "rightEyeLower0",
            "rightEyeUpper1",
            "rightEyeLower1",
            "rightEyeUpper2",
            "rightEyeLower2",
            "rightEyeUpper3",
            "rightEyeLower3",
            "rightEyebrowUpper",
            "rightEyebrowLower",
            // "rightEyeIris",
            "leftEyeUpper0",
            "leftEyeLower0",
            "leftEyeUpper1",
            "leftEyeLower1",
            "leftEyeUpper2",
            "leftEyeLower2",
            "leftEyeUpper3",
            "leftEyeLower3",
            "leftEyebrowUpper",
            "leftEyebrowLower",
            // "leftEyeIris",
            "midwayBetweenEyes",
            "noseTip",
            "noseBottom",
            "noseRightCorner",
            "noseLeftCorner",
            "rightCheek",
            "leftCheek"
        };
    }

    // Function to get keypoints by feature name
    const std::vector<int>& getFaceKeypoints(const std::string& feature) const {
        if (dict_TFLite.find(feature) != dict_TFLite.end()) {
            return dict_TFLite.at(feature);
        }
        else {
            throw std::invalid_argument("Bad access request to TFLite Facial Features: "
                                        "Feature <" + feature + "> not found.");
        }
    }
};

#endif // FACEMESHKEYPOINTS_H
