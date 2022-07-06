import cv2
import numpy as np
image_path = "Data/train/0--Parade/0_Parade_marchingband_1_849.jpg"

image = cv2.imread(image_path)

# 449 330 122 149 453 365 453 383 455 400 457 416 461 433 470 449 480 463 494 474 510 478 526 476 538 466 548 453 556 438 562 423 566 407 570 392 572 375 467 362 476 357 487 357 497 360 507 364 532 364 541 361 551 359 560 361 566 368 518 375 517 386 517 397 517 408 502 412 508 415 515 418 522 416 528 414 480 373 487 370 494 371 501 375 494 376 487 375 532 377 539 374 547 374 552 378 546 379 539 379 486 427 496 425 506 425 514 428 522 426 531 428 539 431 531 443 521 448 513 449 505 447 495 440 490 428 506 430 513 431 521 431 536 432 521 439 513 440 506 438

string = "449 330 122 149 453 365 453 383 455 400 457 416 461 433 470 449 480 463 494 474 510 478 526 476 538 466 548 453 556 438 562 423 566 407 570 392 572 375 467 362 476 357 487 357 497 360 507 364 532 364 541 361 551 359 560 361 566 368 518 375 517 386 517 397 517 408 502 412 508 415 515 418 522 416 528 414 480 373 487 370 494 371 501 375 494 376 487 375 532 377 539 374 547 374 552 378 546 379 539 379 486 427 496 425 506 425 514 428 522 426 531 428 539 431 531 443 521 448 513 449 505 447 495 440 490 428 506 430 513 431 521 431 536 432 521 439 513 440 506 438"

temp = string.split()

x, y, w, h = int(temp[0]), int(temp[1]), int(temp[2]), int(temp[3])

points1 = np.array([(0, 0),
                   (image.shape[1] - 1, 0),
                   (image.shape[1] - 1, image.shape[0] - 1)], np.float32)

points2 = np.array([(0, 0),
                   (255, 0),
                   (255, 255)], np.float32)

resized = cv2.getAffineTransform(points1, points2)

image_transformed = cv2.warpAffine(image, resized, (256, 256))

new_x, new_y, z = [x , y] @ resized

print(new_x, new_y)

mage2 = cv2.circle(image, (x,y), radius=5, color=(0, 0, 255), thickness=-1)
mage2 = cv2.circle(image_transformed, (int(new_x),int(new_y)), radius=5, color=(0, 0, 255), thickness=-1)
cv2.imshow("first", image)
cv2.imshow("second", image_transformed)


cv2.waitKey()