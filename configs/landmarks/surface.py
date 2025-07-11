import numpy as np

body_idx = [
    4,   9,  14,  16,  19,  21,  25,  26,  29,  34,  35,  38,  39,
    40,  46,  47,  48,  57,  63,  64,  67,  69,  70,  71,  72,  73,
    79,  82,  83,  85,  87,  94,  96,  97, 100, 103, 108, 116, 117,
    119, 122, 124, 128, 130, 133, 134, 139, 142, 143, 146, 148, 150,
    152, 153, 154, 155, 156, 159, 160, 162, 163, 168, 169, 170, 176,
    179, 181, 182, 192, 194, 197, 198, 199, 201, 210, 211, 213, 214,
    217, 219, 220, 225, 226, 229, 231, 233, 235, 237, 238, 241, 247,
    248, 251, 253, 254, 260, 265, 270, 272, 275, 277, 281, 282, 285,
    290, 291, 294, 295, 296, 302, 303, 304, 313, 319, 320, 323, 325,
    326, 327, 328, 329, 335, 338, 339, 341, 343, 350, 352, 353, 356,
    359, 364, 372, 373, 375, 378, 380, 384, 386, 389, 390, 395, 398,
    399, 402, 404, 406, 408, 409, 410, 411, 412, 415, 416, 418, 419,
    424, 425, 426, 432, 435, 437, 438, 448, 450, 453, 454, 455, 457,
    466, 467, 469, 470, 473, 475, 476, 481, 482, 485, 487, 489, 491,
    493, 494, 497, 503, 504, 507, 509, 510]
l_hand_idx = [
    3,   5,   7,   8,  10,  12,  18,  20,  27,  37,  41,  42,  43,
    45,  50,  53,  56,  58,  68,  75,  81,  84,  91,  92, 102, 104,
    105, 106, 112, 113, 127, 129, 132, 135, 136, 137, 140, 157, 166,
    167, 177, 178, 180, 184, 185, 186, 188, 190, 193, 195, 207, 208,
    209, 212, 215, 234, 239, 240, 242, 243, 244, 246, 255]
r_hand_idx = [
    259, 261, 263, 264, 266, 268, 274, 276, 283, 293, 297, 298, 299,
    301, 306, 309, 312, 314, 324, 331, 337, 340, 347, 348, 358, 360,
    361, 362, 368, 369, 383, 385, 388, 391, 392, 393, 396, 413, 422,
    423, 433, 434, 436, 440, 441, 442, 444, 446, 449, 451, 463, 464,
    465, 468, 471, 490, 495, 496, 498, 499, 500, 502, 511]
l_feet_idx = [
    2,   6,  17,  23,  24,  28,  33,  49,  51,  55,  59,  61,  74,
    76,  77,  78,  80,  88,  95,  99, 110, 120, 121, 125, 138, 145,
    147, 158, 161, 164, 165, 173, 174, 175, 196, 203, 204, 216, 221,
    222, 224, 230, 232, 249]
r_feet_idx = [
    258, 262, 273, 279, 280, 284, 289, 305, 307, 311, 315, 317, 330,
    332, 333, 334, 336, 344, 351, 355, 366, 376, 377, 381, 394, 401,
    403, 414, 417, 420, 421, 429, 430, 431, 452, 459, 460, 472, 477,
    478, 480, 486, 488, 505]
l_head_idx = [
    0,   1,  11,  13,  15,  22,  30,  31,  32,  36,  44,  52,  54,
    60,  62,  65,  66,  86,  89,  90,  93,  98, 101, 107, 109, 111,
    114, 115, 118, 123, 126, 131, 141, 144, 149, 151, 171, 172, 183,
    187, 189, 191, 200, 202, 205, 206, 218, 223, 227, 228, 236, 245,
    250, 252]
r_head_idx = [
    256, 257, 267, 269, 271, 278, 286, 287, 288, 292, 300, 308, 310,
    316, 318, 321, 322, 342, 345, 346, 349, 354, 357, 363, 365, 367,
    370, 371, 374, 379, 382, 387, 397, 400, 405, 407, 427, 428, 439,
    443, 445, 447, 456, 458, 461, 462, 474, 479, 483, 484, 492, 501,
    506, 508]

body_parts_dict = {
    "body": body_idx,
    "left_hand": l_hand_idx,
    "right_hand": r_hand_idx,
    "left_feet": l_feet_idx,
    "right_feet": r_feet_idx,
    "left_head": l_head_idx,
    "right_head": r_head_idx,
}
body_idx = np.array(body_idx + l_hand_idx + r_hand_idx + l_feet_idx + r_feet_idx + l_head_idx + r_head_idx)

flip_pairs = [[0, 256], [1, 257], [2, 258], [3, 259], [4, 260], 
              [5, 261], [6, 262], [7, 263], [8, 264], [9, 265], 
              [10, 266], [11, 267], [12, 268], [13, 269], [14, 270], 
              [15, 271], [16, 272], [17, 273], [18, 274], [19, 275], 
              [20, 276], [21, 277], [22, 278], [23, 279], [24, 280], 
              [25, 281], [26, 282], [27, 283], [28, 284], [29, 285], 
              [30, 286], [31, 287], [32, 288], [33, 289], [34, 290], 
              [35, 291], [36, 292], [37, 293], [38, 294], [39, 295], 
              [40, 296], [41, 297], [42, 298], [43, 299], [44, 300], 
              [45, 301], [46, 302], [47, 303], [48, 304], [49, 305], 
              [50, 306], [51, 307], [52, 308], [53, 309], [54, 310], 
              [55, 311], [56, 312], [57, 313], [58, 314], [59, 315], 
              [60, 316], [61, 317], [62, 318], [63, 319], [64, 320], 
              [65, 321], [66, 322], [67, 323], [68, 324], [69, 325], 
              [70, 326], [71, 327], [72, 328], [73, 329], [74, 330], 
              [75, 331], [76, 332], [77, 333], [78, 334], [79, 335], 
              [80, 336], [81, 337], [82, 338], [83, 339], [84, 340], 
              [85, 341], [86, 342], [87, 343], [88, 344], [89, 345], 
              [90, 346], [91, 347], [92, 348], [93, 349], [94, 350], 
              [95, 351], [96, 352], [97, 353], [98, 354], [99, 355], 
              [100, 356], [101, 357], [102, 358], [103, 359], [104, 360], 
              [105, 361], [106, 362], [107, 363], [108, 364], [109, 365], 
              [110, 366], [111, 367], [112, 368], [113, 369], [114, 370], 
              [115, 371], [116, 372], [117, 373], [118, 374], [119, 375], 
              [120, 376], [121, 377], [122, 378], [123, 379], [124, 380], 
              [125, 381], [126, 382], [127, 383], [128, 384], [129, 385], 
              [130, 386], [131, 387], [132, 388], [133, 389], [134, 390], 
              [135, 391], [136, 392], [137, 393], [138, 394], [139, 395], 
              [140, 396], [141, 397], [142, 398], [143, 399], [144, 400], 
              [145, 401], [146, 402], [147, 403], [148, 404], [149, 405], 
              [150, 406], [151, 407], [152, 408], [153, 409], [154, 410], 
              [155, 411], [156, 412], [157, 413], [158, 414], [159, 415], 
              [160, 416], [161, 417], [162, 418], [163, 419], [164, 420], 
              [165, 421], [166, 422], [167, 423], [168, 424], [169, 425], 
              [170, 426], [171, 427], [172, 428], [173, 429], [174, 430], 
              [175, 431], [176, 432], [177, 433], [178, 434], [179, 435], 
              [180, 436], [181, 437], [182, 438], [183, 439], [184, 440], 
              [185, 441], [186, 442], [187, 443], [188, 444], [189, 445], 
              [190, 446], [191, 447], [192, 448], [193, 449], [194, 450], 
              [195, 451], [196, 452], [197, 453], [198, 454], [199, 455], 
              [200, 456], [201, 457], [202, 458], [203, 459], [204, 460], 
              [205, 461], [206, 462], [207, 463], [208, 464], [209, 465], 
              [210, 466], [211, 467], [212, 468], [213, 469], [214, 470], 
              [215, 471], [216, 472], [217, 473], [218, 474], [219, 475], 
              [220, 476], [221, 477], [222, 478], [223, 479], [224, 480], 
              [225, 481], [226, 482], [227, 483], [228, 484], [229, 485], 
              [230, 486], [231, 487], [232, 488], [233, 489], [234, 490], 
              [235, 491], [236, 492], [237, 493], [238, 494], [239, 495], 
              [240, 496], [241, 497], [242, 498], [243, 499], [244, 500], 
              [245, 501], [246, 502], [247, 503], [248, 504], [249, 505], 
              [250, 506], [251, 507], [252, 508], [253, 509], [254, 510], [255, 511]
]

subsample_pts_fn = "/is/cluster/fast/scratch/hcuevas/bedlam_lab/verts_512.pkl"