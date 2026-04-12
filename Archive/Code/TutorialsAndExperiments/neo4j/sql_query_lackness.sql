-- Person: Id, Person
-- PersonFriend: PersonID, FriendId

-- Who are Bob's friends?

Select p1.Person
From Person p1 JOIN PersonFriend
ON PersonFriend.FriendId =p1.Id
JOIN Person p2 ON PersonFriend.PersonID=p2.Id
Where p2.person= 'Bob'
-- First join for putting names of the friends into join table
-- Second join is with id = personId and take it for 'Bob'

-- Who is friends with Bob?
-- Now Bob needs to be target , source are other individuals
Select p1.Person
From Person p1 JOIN PersonFriend
ON PersonFriend.PersonId= p1.Id
JOIN Person p2 ON PersonFriend.FriendId = p2.Id
Where p2.Person= 'Bob'
-- First join is for putting names of the sources into joined table
-- Second join is for putting names of the targets and obtain the rows for Bob.

-- Who are the Alice's friends-of-friends

Select p1.Person AS PERSON, p2.Person AS
FRIEND_OF_FRIEND FROM PersonFriend pf1 JOIN Person p1
ON pf1.PersonId =p1.Id
JOIN PersonFriend pf2
ON pf2.PErsonId= pf1.FriendId
JOIN Person p2
ON pf2.FriendId= p2.Id
Where p1.Person= 'Alice' AND pf2.FriendId <> p1.Id

-- join p1 to pf1 for making source "Alice"
-- join pf1 to pf2 for Alice's friends of friends
-- join pf2 to p2 to take the names of the friends of friends
-- exclude Friend Circle with  pf2.FriendId <> p1.Id (e.g. Alice->Bob->Alice)


-- Relational logic does not scale (slower and slower) and it is getting worse as complexity is increasing (joining more and more).
-- It does not store relationship.



-- NOSQL Databases also lack relationships for recommendation systems
-- User : Alice, friends :[Bob] - User: Zach, friends: [Alice,John]..
-- Finding friends of a person is easy, but finding who are considered a person as friend is slower.


-- Graph Databases embrace relationships with keeping track of them on edges.

-- As depth increases, RDBMS execution times goes to infinity, neo4j can execute
-- them in reasonable time.
