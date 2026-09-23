# Design Decisions

## Goal
Build and understand a chest X-ray classification baseline for NORMAL vs PNEUMONIA.

## Why VGG16?
VGG16 is a transfer-learning baseline because it is well understood, easy to inspect, and useful for learning transfer learning. It is not claimed to be optimal, state-of-the-art, or production-grade for medical imaging.

## Why transfer learning?
Reuse general visual features from pretrained ImageNet weights instead of training the whole visual representation from scratch.

## Why freeze the backbone initially?
Start with a simple, fast baseline by keeping pretrained convolutional layers fixed and training the classification head first. Fine-tuning is a later experiment.

## Patient-level splitting
Split patients, not images. A patient must appear in exactly one of train, validation, or test. This prevents the same patient's images from crossing split boundaries and creating leakage.

Target split: 70% train, 15% validation, 15% test, at patient level.

Before training, verify there is zero patient overlap and record split statistics.

## Augmentation
Add augmentation only after the baseline pipeline is correct. Every transform must be checked for whether it preserves diagnostic meaning.

## Scope
This is an educational ML baseline and does not establish clinical validity, safety, or deployment readiness.
