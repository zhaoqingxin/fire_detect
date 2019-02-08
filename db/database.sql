CREATE DATABASE  IF NOT EXISTS `phoenix`;
USE `phoenix`;

DROP TABLE IF EXISTS `predict`;
CREATE TABLE `phoenix`.`predict` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(255) NOT NULL,
  `classify` INT NOT NULL,
  `probabilities` FLOAT NOT NULL,
  `created` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`));

DROP TABLE IF EXISTS `evaluate`;
CREATE TABLE `phoenix`.`evaluate` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `accuracy` FLOAT NOT NULL,
  `loss` FLOAT NOT NULL COMMENT '损失函数值',
  `global_step` INT NOT NULL COMMENT '采用的参数是训练多少步的',
  `created` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`));

DROP TABLE IF EXISTS `train`;
CREATE TABLE `phoenix`.`train` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `train_image_count` INT NOT NULL,
  `train_step` INT NOT NULL COMMENT '本次训练的总步数',
  `epoch` INT NOT NULL,
  `duration` VARCHAR(45) NOT NULL,
  `created` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`));
