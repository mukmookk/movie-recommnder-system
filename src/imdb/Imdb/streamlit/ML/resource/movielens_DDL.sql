-- movie
-- Initialize table structure
--
DROP TABLE IF EXISTS `ratings`;
DROP TABLE IF EXISTS `user`;
DROP TABLE IF EXISTS `movie_genres`;
DROP TABLE IF EXISTS `movie`;



--
-- Create table structure
--
CREATE TABLE `movie` (
  `movieId` int NOT NULL,
  `movieTitle` varchar(100) NOT NULL,
  `releaseDate` text NOT NULL,
  `videoReleaseDate` text NOT NULL,
  `IMDbURL` text,
  PRIMARY KEY (`movieId`)
);

CREATE TABLE `movie_genres` (
  `mgenreId` int NOT NULL AUTO_INCREMENT,
  `movieId` int NOT NULL,
  `genre` varchar(100) NOT NULL,
  PRIMARY KEY (`mgenreId`),
  KEY `mvoieId_idx` (`movieId`),
  CONSTRAINT `mvoieId` FOREIGN KEY (`movieId`) REFERENCES `movie` (`movieId`)
) AUTO_INCREMENT=2894;

CREATE TABLE `user` (
  `userId` int NOT NULL,
  `age` int NOT NULL,
  `gender` char(1) NOT NULL,
  `occupation` varchar(100) NOT NULL,
  `ZIPCODE` varchar(10) NOT NULL,
  PRIMARY KEY (`userId`)
);

CREATE TABLE `ratings` (
  `ratingId` int NOT NULL AUTO_INCREMENT,
  `userId` int NOT NULL,
  `movieId` int NOT NULL,
  `ratingScore` int NOT NULL,
  `timestamp` text NOT NULL,
  PRIMARY KEY (`ratingId`),
  KEY `movieId_idx` (`movieId`),
  KEY `userId_idx` (`userId`),
  CONSTRAINT `movieId` FOREIGN KEY (`movieId`) REFERENCES `movie` (`movieId`),
  CONSTRAINT `userId` FOREIGN KEY (`userId`) REFERENCES `user` (`userId`)
) AUTO_INCREMENT=300458;