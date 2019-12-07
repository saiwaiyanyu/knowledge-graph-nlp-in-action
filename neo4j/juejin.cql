// 创建主键
CREATE CONSTRAINT ON (n:User) ASSERT n.id IS UNIQUE;


// 导入节点
USING PERIODIC COMMIT 5000
LOAD CSV FROM 'file:///data/basicinfo.csv' AS row FIELDTERMINATOR '\t'
WITH row 
WITH apoc.convert.fromJsonMap( row[0]) AS info WHERE LENGTH(info["d"][info["id"]]) = 1
MERGE (u:User{id:info["id"]})
ON CREATE SET u += apoc.map.clean(info["d"][info["id"]], ['community','roles','cancelBakup'], []) 
;


// 导入关系
USING PERIODIC COMMIT 5000
LOAD CSV FROM 'file:///data/data.csv' AS row FIELDTERMINATOR '\t'
WITH row
WITH apoc.convert.fromJsonMap(apoc.text.join(row,",")) AS info 
UNWIND info["d"] AS elem
WITH  
info["type"] AS type,
CASE WHEN info["type"] = "Followee" THEN  elem["followerId"] WHEN info["type"] = "Follower" THEN  elem["follower"]["objectId"] END AS startId,
CASE WHEN info["type"] = "Followee" THEN elem["followee"]["objectId"] WHEN info["type"] = "Follower" THEN  elem["followeeId"] END AS endId ,
elem["createdAtString"] AS createdAtString,
elem["updatedAtString"] AS updatedAtString
MATCH (startNode:User{id:startId}),(endNode:User{id:endId})
MERGE(startNode)-[r:follow]->(endNode)
  ON CREATE SET r.createdAtString = createdAtString,
                r.updatedAtString = updatedAtString
;




// 每月新增用户和关系情况
MATCH (u) WHERE EXISTS(u.createdAt)
WITH apoc.date.format(apoc.date.fromISO8601(u.createdAt),'ms','yyyy-MM') AS createdAt, COUNT(1) AS num
WITH apoc.map.fromLists(COLLECT(createdAt), COLLECT(num)) AS userMap
MATCH ()<-[r]-() 
WITH userMap,r.createdAtString AS createdAtString
WITH userMap,apoc.date.format(apoc.date.fromISO8601(createdAtString),'ms','yyyy-MM') AS createdAtString, COUNT(1) AS num
WITH createdAtString, num AS followNum  ,userMap[createdAtString] AS userNum
ORDER BY createdAtString
WITH COLLECT({createdAtString:createdAtString, userNum:userNum ,followNum:followNum} ) AS infoList
UNWIND RANGE(0,LENGTH(infoList)-1) AS index
WITH infoList[index]["createdAtString"] AS createdAtString,
infoList[index]["userNum"] AS userNum,
infoList[index]["followNum"] AS followNum,
apoc.coll.sum([elem IN infoList[..index+1] | elem["userNum"]]) AS userNumSum,
apoc.coll.sum([elem IN infoList[..index+1] | elem["followNum"]]) AS followNumSum
RETURN createdAtString,userNum,followNum,userNumSum,followNumSum,followNumSum/userNumSum AS followPerUser
ORDER BY createdAtString DESC
LIMIT 10
;


// 粉丝数、最早注册等统计, 每日新增粉丝
MATCH (u:User)<-[r]-() WHERE EXISTS(u.createdAt) AND EXISTS(r.createdAtString)
WITH u,r.createdAtString AS createdAtString
WITH u,apoc.date.format(apoc.date.fromISO8601(createdAtString),'ms','yyyy-MM-dd') AS createdAtString, COUNT(1) AS num
WITH u, MAX(num ) AS maxNum
RETURN u.username,
apoc.date.format(apoc.date.fromISO8601(u.createdAt),'ms','yyyy-MM-dd') AS createdAt,
apoc.date.convert( timestamp()+3600*8*1000- apoc.date.fromISO8601(u.createdAt), 'ms', 'd') AS days,
u.followersCount AS followersCount,
u.followersCount/apoc.date.convert( timestamp()+3600*8*1000- apoc.date.fromISO8601(u.createdAt), 'ms', 'd') AS aveNum,
maxNum 
// ORDER BY createdAt 
ORDER BY aveNum DESC
// ORDER BY maxNum DESC
LIMIT 10
;



// 粉丝分布
MATCH (u:User) 
WHERE EXISTS(u.followersCount)
RETURN COUNT(u.followersCount) AS count,
AVG(u.followersCount) AS ave, 
percentileDisc(u.followersCount, 0.5) AS `50%`, 
percentileDisc(u.followersCount, 0.75) AS `75%`, 
percentileDisc(u.followersCount, 0.90) AS `90%`, 
percentileDisc(u.followersCount, 0.95) AS `95%`, 
percentileDisc(u.followersCount, 0.99) AS `99%`, 
percentileDisc(u.followersCount, 0.999) AS `99.9%`, 
percentileDisc(u.followersCount, 0.9999) AS `99.99%`, 
percentileDisc(u.followersCount, 0.99999) AS `99.999%`, 
percentileDisc(u.followersCount, 1) AS `100%`
;


// 粉丝占比
MATCH (u:User) 
WHERE EXISTS(u.followersCount) 
WITH 
SUM(CASE WHEN u.followersCount <=60 THEN u.followersCount ELSE 0 END) AS low,
SUM(CASE WHEN u.followersCount >60 THEN u.followersCount ELSE 0 END) AS high,
SUM(u.followersCount) AS followersCountSum
RETURN low,high,followersCountSum,
low*1.0/followersCountSum AS lowPercent,
high*1.0/followersCountSum AS highPercent
;



// pageRank 影响力挖掘分析
MATCH(u0:User{id:'551d677ee4b0cd5b623f49cb'})
CALL algo.pageRank.stream('User', 'follow', {iterations:20, dampingFactor:0.85}) YIELD nodeId, score
WITH u0,algo.getNodeById(nodeId) AS u, score
WITH u,score, CASE WHEN EXISTS((u0)-[:follow]-(u)) THEN 1 ELSE 0 END AS hasRelation
RETURN 
u.username AS username,
score,hasRelation
ORDER BY score DESC 
LIMIT 20
;



// pageRank 影响力前10 的用户之间的关系

CALL algo.pageRank.stream('User', 'follow', {iterations:20, dampingFactor:0.85}) YIELD nodeId, score
WITH algo.getNodeById(nodeId) AS u, score
WITH u,score ORDER BY  score DESC  LIMIT 10
WITH COLLECT(u) AS Users
UNWIND Users AS u1
UNWIND Users AS u2
MATCH p=(u1)-[r:follow]-(u2)
RETURN p;



// 互粉的情况
MATCH (u1:User)<-[:follow]-(u2:User) WHERE id(u1)>id(u2)
MATCH (u1)-[:follow]->(u2)
RETURN COUNT(DISTINCT u1.id+u2.id) AS num
;



// 自我互粉的情况
MATCH (u1:User)<-[:follow]-(u2:User) WHERE id(u1)=id(u2)
RETURN COUNT(DISTINCT u1.id) AS num
;


// 互粉最多的用户
MATCH (u1:User)<-[:follow]-(u2:User)// WHERE id(u1)>id(u2)
MATCH (u1)-[:follow]->(u2)
WITH u1,u2
WITH u1,COUNT(u2) AS num
RETURN  u1.username,num
ORDER BY num DESC
LIMIT 20
;



// 最长路径
MATCH path = (u1:User)<-[:follow*9..9]-(u2:User) 
WHERE ALL(u IN  [u0 IN NODES(path)| size([m IN FILTER(f IN NODES(path) WHERE ID(f)=ID(u0))])] WHERE u<2) 
RETURN path, LENGTH(path)
// ORDER BY LENGTH(path) DESC 
LIMIT 1
;



// 发布文章最多的，发布沸点最多的
MATCH (u:User)  WHERE
// EXISTS(u.pinCount) 
EXISTS(u.postedPostsCount) 
RETURN u.id AS id,
u.username AS username,
u.pinCount AS pinCount,
u.pinCount/apoc.date.convert( timestamp()+3600*8*1000- apoc.date.fromISO8601(u.createdAt), 'ms', 'd')  AS pinCountAve,
u.postedPostsCount/apoc.date.convert( timestamp()+3600*8*1000- apoc.date.fromISO8601(u.createdAt), 'ms', 'd')  AS postedPostsCountAve,
u.totalCollectionsCount AS totalCollectionsCount,
u.postedPostsCount AS postedPostsCount,
u.totalViewsCount AS   totalViewsCount
ORDER BY postedPostsCount DESC 
LIMIT 10
;

