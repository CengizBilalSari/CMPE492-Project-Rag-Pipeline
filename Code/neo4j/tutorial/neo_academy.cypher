//Return node which has Person label and name property as 'Tom Cruise'
MATCH (n:Person)
WHERE n.name = 'Tom Cruise'
RETURN n;

// return movie node that has 'Toy Story' title and relationship of it with person nodes.
MATCH (m:Movie)<-[r:ACTED_IN]-(p:Person)
WHERE m.title = 'Toy Story'
RETURN m, r, p;

// to see the genres relationship of the Toy Story
MATCH (m:Movie)-[r:IN_GENRE]-(g:Genre)
WHERE m.title= 'Toy Story'
Return m,r,g;


//return tabular data by including the properties of the nodes.
MATCH (m:Movie)-[r:IN_GENRE]->(g:Genre)
WHERE m.title = 'Toy Story'
RETURN m.title, g.name;
 
// to return tabular data for a 50 example of rating of a movie
Match (u:User)-[r:RATED]->(m:Movie)
LIMIT 50
Return u.name,r.rating,m.title;



// return the individuals that act with Tom Cruise at the same movie and put his/her role in the tabular data too.
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)<-[r:ACTED_IN]-(p2:Person)
WHERE p.name = 'Tom Cruise'
RETURN p2.name AS actor, m.title AS movie, r.role AS role;

 
// Create a pattern with a Merge clause , it will be created if it is not already 
MERGE (m:Movie {title: "Arthur the King"})
SET m.year = 2024
RETURN m;


// Create rating for the movie for User with the name "Cengiz", if I PUT  name property with SET (not in merge)
// It gets all of the users then set their name to Cengiz, be careful about it!!
MERGE (m:Movie {title: "Arthur the King"})
MERGE (u:User {name: "Cengiz"})
MERGE (u)-[r:RATED {rating: 5}]->(m)
RETURN u, r, m;

// to see the ontology of the db
call db.schema.visualization;


// Convention: 
// Node Labels	        CamelCase	        :FinancialAccount
// Relationship Types	UPPER_SNAKE_CASE	:LIVES_IN
// Property Keys	    camelCase	        :accountBalance
// Variables	        camelCase	        :myNode
// Keywords	            UPPERCASE	        :MATCH, MERGE

// we can give range for the where clause
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE p.name="Tom Hanks" and  "2012">m.released >"2003"
RETURN p.name, m.title, m.released;

//Cypher has a set of string-related keywords that you can use in your WHERE clauses to test string property values. 
// We can specify STARTS WITH, ENDS WITH, and CONTAINS e.g.:
MATCH (p:Person)-[:ACTED_IN]->()
WHERE p.name STARTS WITH 'Michael'
RETURN p.name;
// Cypher is case sensitive so using toLower() or toUpper() is a better practice.ALTER

MATCH (p:Person)-[:ACTED_IN]->()
WHERE toLower(p.name) STARTS WITH 'michael'
RETURN p.name;



// people who wrote a movie but not directed it
MATCH (p:Person)-[:WROTE]->(m:Movie)
WHERE NOT exists( (p)-[:DIRECTED]->(m) )
RETURN p.name, m.title;


//  person and its roles in the movie who plays the Neo Role in the matrix 
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
WHERE  'Neo' IN r.roles AND m.title='The Matrix'
RETURN p.name, r.roles;


// to check is there any roles list with a size of more than 1 and see it in decreasing order
MATCH (p:Person)-[rel:ACTED_IN]->(m:Movie)
WHERE size(rel.roles) > 1
RETURN p.name AS actor, m.title AS movie, rel.roles AS roles
ORDER BY size(rel.roles) DESC;


// see the keys in the graph
CALL db.propertyKeys();
// see the keys of a person for each person node.
MATCH (p:Person)
RETURN p.name, keys(p);


//Cypher has a CREATE clause you can use for creating nodes. The benefit of using CREATE is that it does not look up the primary key before adding the node. 
// We can use CREATE if we  are sure our data is clean and we want greater speed during import. 
//We use MERGE in this training because it eliminates duplication of nodes.

// For merge of the relationship, if we do not specify the direction, left-to-right is the default.






// Remove a property
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
WHERE p.name = 'Michael Caine' AND m.title = 'The Dark Knight'
REMOVE r.roles
RETURN p, r, m;

MATCH (p:Person)
WHERE p.name = 'Gene Hackman'
SET p.born = null
RETURN p;




// set property according to whether it is create or match
MERGE (p:Person {name: 'McKenna Grace'})
ON CREATE SET p.createdAt = datetime()
ON MATCH SET p.updatedAt = datetime()
SET p.born = 2006
RETURN p;

// By default, we cannot delete a node if it has a relationship. Use detach delete to delete both node and its relationships.
MATCH (p:Person {name: 'Jane Doe'})
DETACH DELETE p;
