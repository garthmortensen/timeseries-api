-- ddl schema
create table cats (
    id serial primary key,
    name varchar(255) not null,
    age integer,
    loved boolean default true,
    created_at timestamp default current_timestamp
);
